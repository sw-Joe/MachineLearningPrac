import os
import glob
import json
from pathlib import Path # Path 객체 사용을 위해 추가

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast

from ml_core.metric import MetricTracker, ClassificationTracker



class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device, ema=None, config=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.ema = ema
        self.config = config
        
        # DDP 상태 파악
        self.is_dist = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.is_master = (self.rank == 0)
        
        # 트래커 초기화
        self.train_tracker = MetricTracker(topk=(1, 5), device=device)
        self.val_tracker = MetricTracker(topk=(1, 5), device=device)
        self.clf_metric = ClassificationTracker(classes=config.dataset.label)
        self.best_val_loss = float("inf")


    def fit(self, train_loader, val_loader, n_epochs, run_id, save_path):
        """전체 학습 루프를 조율"""
        for epoch in range(n_epochs):
            # DDP 샘플러 셔플 동기화
            if self.is_dist and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # 1. 에포크 실행
            self._train_epoch(train_loader, run_id)
            misclassified = self._validate_epoch(val_loader)

            # 2. 지표 동기화 (DDP 합산)
            self.train_tracker.synchronize()
            self.val_tracker.synchronize()

            # 3. 종료 처리 (로깅, 저장, 스케줄러)
            if self.is_master:
                self._handle_epoch_end(epoch, n_epochs, run_id, save_path, misclassified)
            
            # 다음 에포크를 위한 초기화
            self.train_tracker.reset()
            self.val_tracker.reset()
            torch.cuda.empty_cache()


    def _train_epoch(self, loader, run_id):
        """Step Logging"""
        self.model.train()
        for x, y, _ in loader:
            x, y = x.to(self.device, non_blocking=True).bfloat16(), y.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicts = self.model(x)
                loss = self.criterion(predicts, y)

            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            self.optimizer.step()
            if self.ema != None:
                self.ema.update()

            # 현재 학습률 추출 (optimizer의 첫 번째 파라미터 그룹 기준)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_tracker.update(loss.item(), predicts, y)
            
            '''Step Logging'''
            if self.is_master and run_id:
                run_id.log({
                    "train/lr": current_lr, # 시각적으로 확인하기 위해 추가
                    # "train/step_loss": loss.item(),
                    # "train/step_acc": (predicts.argmax(1) == y).float().mean().item(),
                })


    @torch.no_grad()
    def _validate_epoch(self, loader):
        self.model.eval()
        self.clf_metric.reset()    # 검증 시작 시 초기화
        if self.ema != None:
            self.ema.apply_shadow()
        local_misclassified = []

        for x, y, paths in loader:
            x, y = x.to(self.device, non_blocking=True).bfloat16(), y.to(self.device, non_blocking=True)
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicts = self.model(x)
                # 비용 함수 계산 추가
                loss = self.criterion(predicts, y)
                # [추가] F1 계산을 위한 배치 데이터 업데이트
                self.clf_metric.update_batch(predicts, y)

            # 트래커 업데이트 로직 복구
            self.val_tracker.update(loss.item(), predicts, y)
            self._collect_misclassified(predicts, y, paths, local_misclassified)
        
        # --- [핵심] 모든 GPU의 리스트를 하나로 통합 ---
        if self.is_dist:
            # 1. 모든 Rank의 리스트를 담을 저장소 준비
            world_size = dist.get_world_size()
            gathered_list = [None] * world_size 
            
            # 2. 모든 GPU의 local_misclassified를 모음 (Pickle 방식)
            dist.all_gather_object(gathered_list, local_misclassified)
            
            # 3. 마스터 노드에서만 리스트를 하나로 병합
            if self.is_master:
                # [[rank0_list], [rank1_list]] -> [rank0 + rank1 combined]
                total_misclassified = [item for sublist in gathered_list for item in sublist]
            else:
                total_misclassified = []
        else:
            total_misclassified = local_misclassified
        if self.ema != None:
            self.ema.restore()
        return total_misclassified


    def _handle_epoch_end(self, epoch, n_epochs, run_id, save_path, misclassified):
        """Epoch Logging, 최적모델 저장"""
        # WandB 차트 정렬을 위한 metric 정의 (첫 에포크에만 실행)
        if epoch == 0 and run_id:
            run_id.define_metric("train/epoch_*", step_metric="epoch")
            run_id.define_metric("validation/epoch_*", step_metric="epoch")

        # 최적 모델 판단(loss 기준 판단)
        if self.val_tracker.avg_loss < self.best_val_loss:
            self.best_val_loss = self.val_tracker.avg_loss
            self._save_checkpoint(save_path, epoch)
            self._save_misclassified(save_path, epoch, misclassified)

        if self.scheduler:
            self.scheduler.step()

        # [추가] F1-Score 및 상세 리포트 산출 (마스터 노드에서만 실행)
        f1_macro = self.clf_metric.get_f1_score(average='macro')
        report = self.clf_metric.get_report()
            
        # Epoch Summary
        print(f'Epoch: {epoch+1:03d}/{n_epochs} | '
            f'Train Loss: {self.train_tracker.avg_loss:.4f} | '
            f'Train Acc: {self.train_tracker.accuracy*100:.2f}% | '
            f'Val Loss: {self.val_tracker.avg_loss:.4f} | '
            f'Val Acc: {self.val_tracker.accuracy*100:.2f}%')
        
        # 클래스별 상세 리포트 출력
        print(f"\n[Classification Report]\n{report}")

        # Epoch Logging
        if run_id:
            run_id.log({
                "train/epoch_acc": self.train_tracker.accuracy,
                "train/epoch_loss": self.train_tracker.avg_loss,
                "train/top1_err": self.train_tracker.get_error_rate(1),
                "train/top5_err": self.train_tracker.get_error_rate(5),
                "validation/epoch_acc": self.val_tracker.accuracy,
                "validation/epoch_loss": self.val_tracker.avg_loss,
                "validation/top1_err": self.val_tracker.get_error_rate(1),
                "validation/top5_err": self.val_tracker.get_error_rate(5),
                "validation/epoch_f1_macro": f1_macro,
                "epoch": epoch + 1
            })


    def _collect_misclassified(self, predicts, targets, paths, storage_list):
        """
        오분류된 샘플의 메타데이터를 수집합니다.
        
        Args:
            predicts (Tensor): 모델의 raw output (logits)
            targets (Tensor): 실제 정답 라벨
            paths (list): 이미지 파일 경로들의 리스트
            storage_list (list): 정보를 저장할 리스트 객체
        """
        # 1. 예측값에서 가장 높은 확률의 인덱스 추출
        preds = predicts.argmax(dim=1)
        
        # 2. 확신도(Confidence) 분석을 위해 Softmax 적용
        probs = predicts.softmax(dim=1)
        
        # 3. 정답과 예측이 다른 인덱스만 마스킹하여 추출
        wrong_indices = (preds != targets).nonzero(as_tuple=True)[0]
        
        # 4. 각 오분류 샘플의 상세 정보를 딕셔너리 형태로 저장
        for idx in wrong_indices:
            storage_list.append({
                "file_path": paths[idx],
                "true_label": targets[idx].item(),
                "pred_label": preds[idx].item(),
                "confidence": probs[idx][preds[idx]].half.item(), # 모델이 얼마나 확신했는지 기록
            })    # half(): float16, float(): float32


    def _save_checkpoint(self, save_path, epoch):
        """
        최적의 모델 상태(Weights)를 파일명에 에포크를 명시하여 저장합니다.
        """
        # 1. 이전 에포크의 pt 파일 검색 및 삭제 (클리닝)
        old_checkpoints = glob.glob(str(save_path / "best_model_ep*.pt"))
        for f in old_checkpoints:
            try:
                os.remove(f)
            except OSError:
                pass

        # 2. 새로운 파일명 정의
        epoch_str = f"ep{epoch + 1:03d}"
        model_name = f"best_model_{epoch_str}.pt"

        # 3. DDP 래핑 여부에 따른 가중치 추출
        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        # 4. 물리적 저장
        torch.save(state_dict, save_path / model_name)
        print(f"   >>> Model Checkpoint Saved: {model_name}")


    def _save_misclassified(self, save_path, epoch, misclassified):
        """
        오분류된 샘플들의 상세 정보를 JSON 파일로 저장합니다.
        """
        # 1. 이전 에포크의 json 파일 검색 및 삭제
        old_jsons = glob.glob(str(save_path / "misclassified_ep*.json"))
        for f in old_jsons:
            try:
                os.remove(f)
            except OSError:
                pass

        # 2. 새로운 파일명 정의
        epoch_str = f"ep{epoch + 1:03d}"
        json_name = f"misclassified_{epoch_str}.json"

        # 3. JSON 직렬화 및 저장
        with open(save_path / json_name, "w", encoding="utf-8") as f:
            json.dump(misclassified, f, indent=4)
            
        print(f"   >>> Misclassified Data Saved: {json_name}")