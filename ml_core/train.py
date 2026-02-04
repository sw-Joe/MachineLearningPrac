"""
    CPU-GPU 통신 오버헤드 줄이기 위해 tensor.item(), tensor.tolist() 사용 줄이기 - Synchronization
    item()으로 선언된 부분을 다른 방법으로 대체
    또는 item()으로 하되 또 다른 방법
"""

import json

from torch import no_grad, bfloat16
import torch.cuda
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import torch.distributed as dist 
import wandb

from metric import MetricTracker
from logger import count_time



class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, current_loss):
        # 유의미한 개선이 있는 경우
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            return True   # best model 갱신 flag
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


@count_time
def fit(device, model, optimizer, scheduler, criterion, trainset_loader, valset_loader, 
          n_epoch: int, run, time, path, ema) -> None:
    """
    Model training loop
    @param: model, optimizer, criterion, trainset_loader, n_epoch: int\n
    """
    # early_stopping = EarlyStopping(
    #     patience=10,
    #     min_delta=1e-4
    # )

    iter: int = 0
    best_val_loss: float = float("inf")

    misclassified_samples = []

    # Autocast & Gradscaler
    try:
        scaler = GradScaler(device=device)
    except:
        scaler = GradScaler()
    
    # 트래커 초기화
    train_tracker = MetricTracker(topk=(1, 5), device=device)
    val_tracker = MetricTracker(topk=(1, 5), device=device)

    # Wandb 기록 on/off
    logging_flag: bool = run is not None and dist.get_rank() == 0

    if logging_flag:
        wandb.define_metric("train/epoch_*", step_metric="epoch")
        wandb.define_metric("validation/epoch_*", step_metric="epoch")
        wandb.define_metric("train/step_*")

    for epoch in range(n_epoch):
        # 매 Epoch 시작 시 초기화
        train_tracker.reset()
        val_tracker.reset()

        ''' [Train Loop] '''
        model.train()

        for x, y, _ in trainset_loader:

            # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
            x_train, y_train = x.to(device).bfloat16(), y.to(device)
            # x_train = x.to(device).bfloat16()
            # y_train = y.to(device)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicts = model(x_train)               # 1. 예측
                loss = criterion(predicts, y_train)     # 2. 비용 함수

            optimizer.zero_grad()                   # 3. gradient 초기화
            loss.backward()                         # 4. backward propagation
            
            # Gradient Clipping: 수치 폭발(NaN) 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()                        # 5. weight 업데이트

            ema.update()

            # 트래커에 배치 결과 기록
            train_tracker.update(loss.item(), predicts, y_train)

            if logging_flag:
                run.log({
                    "train/step_loss": loss.item(),
                    "train/step_acc": (predicts.argmax(1) == y_train).float().mean().item(),
                    "train/top1_err": train_tracker.get_error_rate(1),
                    "train/top5_err": train_tracker.get_error_rate(5),
                })
            # preds = predicts.argmax(dim=1)              # 각 샘플에 대해 가장 큰 logit을 가진 클래스 인덱스 반환
            # correct = (preds == y_train).sum().item()   # 이번 배치에서 맞춘 갯수

            iter += 1

        # Train 루프 종료 후 모든 GPU 결과 통합
        train_tracker.synchronize()


        ''' [Validation Loop] '''
        model.eval()
        epoch_misclassified = []

        with torch.no_grad():
            ema.apply_shadow() # 검증 시 EMA 가중치 적용

            for x, y, paths in valset_loader:
                x_val, y_val = x.to(device).bfloat16(), y.to(device)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    predicts = model(x_val)
                    loss = criterion(predicts, y_val)

                preds = predicts.argmax(dim=1)

                # 추가: 기존 predicts 변수에서 바로 확률값 추출
                # Logit을 Softmax로 변환하여 0~1 사이의 확신도로 만듭니다.
                probs = predicts.softmax(dim=1)

                wrong_indices = (preds != y_val).nonzero(as_tuple=True)[0]

                for idx in wrong_indices:
                    epoch_misclassified.append({
                        "file_path": paths[idx],
                        "true_label": y_val[idx].item(),
                        "pred_label": preds[idx].item(),
                        # 3. 계산된 probs에서 예측한 클래스의 확률값만 추출하여 추가
                        "confidence": probs[idx][preds[idx]].item(),
                        "epoch": epoch + 1
                    })
                
                # 트래커에 배치 결과 기록
                val_tracker.update(loss.item(), predicts, y_val)

            # Validation 루프 종료 후 모든 GPU 결과 통합
            val_tracker.synchronize()

        # 에폭 결과 로깅 (Property 사용으로 계산 간소화)
        if logging_flag:
            run.log({
                "train/epoch_loss": train_tracker.avg_loss,
                "train/epoch_acc": train_tracker.accuracy,
                "validation/epoch_loss": val_tracker.avg_loss,
                "validation/epoch_acc": val_tracker.accuracy,
                "validation/top1_err": val_tracker.get_error_rate(1),
                "validation/top5_err": val_tracker.get_error_rate(5),
                "epoch": epoch + 1,
            })

        ## Early Stopping
        # is_best = early_stopping.step(epoch_val_loss_avg)

        # if is_best:
        #     torch.save(model.state_dict(), f"./{path}{time}/best_model_{time}.pt")

        # if early_stopping.should_stop:
        #     print(
        #         f"Early stopping triggered at epoch {epoch+1} | "
        #         f"best val loss: {early_stopping.best_loss:.6f}"
        #     )
        #     break

        # 모델 저장 로직
        if val_tracker.avg_loss < best_val_loss:
            best_val_loss = val_tracker.avg_loss
            misclassified_samples = epoch_misclassified

            ### 개발 예정 ###
            # checkpoint = {
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': val_loss,
            # }

            if dist.get_rank() == 0:
                state_dict = model.module.state_dict() if dist.is_initialized() else model.state_dict()
                torch.save(state_dict, f"{path}{time}/best_model_{time}.pt")
                # torch.save(model.state_dict(), f"{path}{time}/best_model_{time}.pt")
                print(f"--- Model saved at epoch {epoch+1} (Loss: {best_val_loss:.6f}) ---")

            # 오분류 결과 저장
            with open(f"{path}{time}/misclassified.json", "w", encoding="utf-8") as f:
                json.dump(misclassified_samples, f, indent=4)
        

        ema.restore()

        # 스케줄러 업데이트
        if scheduler is not None:
            scheduler.step()
        
        # 학습률 로깅
        current_lr = optimizer.param_groups[0]['lr']

        if dist.get_rank() == 0:
            print(f'Epoch: {epoch+1:03d}/{n_epoch} | LR: {current_lr:.6f} | '
                f'Train Loss: {train_tracker.avg_loss:.4f} | Val Loss: {val_tracker.avg_loss:.4f} | '
                f'Val Acc: {val_tracker.accuracy*100:.2f}%')

        # train.py의 에폭 루프 끝부분이나 시작 부분
        torch.cuda.empty_cache()


    if run is not None:
        run.finish()