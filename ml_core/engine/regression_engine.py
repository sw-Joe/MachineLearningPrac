import os
import glob

import torch
from torch.amp.autocast_mode import autocast

from ml_core.engine.base import BaseTrainer
from ml_core.metric import RegTracker



class RegTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.best_mae = float("inf")

    def _init_trackers(self):
        """회귀 전용 트래커 초기화"""
        self.train_tracker = RegTracker(device=self.device)
        self.val_tracker = RegTracker(device=self.device)

    def _train_epoch(self, loader, run_id):
        """회귀용 학습: 라벨을 float로 강제 형변환 및 차원 맞춤"""
        self.model.train()
        
        # 오답 추론 이미지를 추출하려면 로더가 img, label, file_name 3개를 반환하도록
        # 프로그래밍해야 함
        for x, y in loader:
            x = x.to(self.device, non_blocking=True).bfloat16()
            y = y.to(self.device, non_blocking=True).float() # 회귀 필수

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(x)
                loss = self.criterion(outputs.view_as(y), y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.ema is not None:
                self.ema.update()

            self.train_tracker.update(loss.item(), outputs, y)

            # Step Logging (WandB)
            if self.is_master and run_id:
                run_id.log({
                    "train/lr": self.optimizer.param_groups[0]['lr']
                })

    @torch.no_grad()
    def _validate_epoch(self, loader):
        """회귀용 검증: MAE/RMSE 모니터링"""
        self.model.eval()
        if self.ema is not None:
            self.ema.apply_shadow()

        for x, y in loader:
            x, y = x.to(self.device).bfloat16(), y.to(self.device).float()
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = self.model(x)
                loss = self.criterion(outputs.view_as(y), y)
            self.val_tracker.update(loss.item(), outputs, y)
        
        if self.ema is not None:
            self.ema.restore()
        return None 

    def _handle_epoch_end(self, epoch, n_epochs, run_id, save_path, val_results):
        """에포크 요약 출력, WandB 로깅, MAE 기준 최적 모델 저장"""
        if self.scheduler:
            self.scheduler.step()

        # 지표 추출
        t_mae = self.train_tracker.mae
        v_mae = self.val_tracker.mae
        v_rmse = self.val_tracker.rmse

        # WandB 차트 정의 (첫 에포크)
        if epoch == 0 and run_id:
            run_id.define_metric("train/epoch_*", step_metric="epoch")
            run_id.define_metric("validation/epoch_*", step_metric="epoch")

        # 콘솔 출력
        print(f'Epoch: {epoch+1:03d}/{n_epochs} | '
              f'Train Loss: {self.train_tracker.avg_loss:.4f} | '
              f'Train MAE: {t_mae:.4f} | '
              f'Val Loss: {self.val_tracker.avg_loss:.4f} | '
              f'Val MAE: {v_mae:.4f} | '
              f'Val RMSE: {v_rmse:.4f}')

        # 최적 모델 판단 및 저장 (MAE 기준)
        if v_mae < self.best_mae:
            self.best_mae = v_mae
            self._save_checkpoint(save_path, epoch)

        # WandB 로깅
        if run_id:
            run_id.log({
                "train/epoch_loss": self.train_tracker.avg_loss,
                "train/epoch_mae": t_mae,
                "validation/epoch_loss": self.val_tracker.avg_loss,
                "validation/epoch_mae": v_mae,
                "validation/epoch_rmse": v_rmse,
                "epoch": epoch + 1
            })

    def _save_checkpoint(self, save_path, epoch):
        """MAE 기준 최적 모델 가중치 저장"""
        # 1. 이전 체크포인트 클리닝
        old_checkpoints = glob.glob(str(save_path / "best_reg_model_ep*.pt"))
        for f in old_checkpoints:
            try:
                os.remove(f)
            except OSError:
                pass

        # 2. 파일명 및 가중치 추출
        epoch_str = f"ep{epoch + 1:03d}"
        model_name = f"best_reg_model_{epoch_str}.pt"
        state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        # 3. 저장
        torch.save(state_dict, save_path / model_name)
        print(f"   >>> Best Model Saved: {model_name} (MAE: {self.best_mae:.4f})")