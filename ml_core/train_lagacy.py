from dataclasses import dataclass, field

from torch import no_grad
import torch.cuda
import wandb

from logger import count_time



""" GPU 존재 확인 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class MetricTracker:
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0

    def _reset(self):
        """매 Epoch마다 변수를 0으로 초기화"""
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0

    @property
    def avg_loss(self):
        # 여기서는 배치의 개수가 아닌 샘플 개수(total) 기준으로 계산
        # 또는 필요에 따라 루프에서 직접 계산 가능
        return self.loss_sum / self.total if self.total > 0 else 0

    @property
    def accuracy(self):
        return (self.correct / self.total) if self.total > 0 else 0
    

train_tracker = MetricTracker()
validation_tracker = MetricTracker()


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


""" 기본 훈련 루프 """
@count_time
def train(model, optimizer, scheduler, criterion, trainset_loader, valset_loader, 
          n_epoch: int, run, time, path) -> None:
    """
    @param: model, optimizer, criterion, trainset_loader, n_epoch: int\n
    (1)예측 → (2)손실 계산 → (3)그래디언트 초기화 → (4)역전파 → (5)가중치 업데이트해보기
    """
    # early_stopping = EarlyStopping(
    #     patience=10,
    #     min_delta=1e-4
    # )
    best_val_loss = float("inf")

    # Wandb 기록 on/off
    if run != None:
        # epoch을 별도의 x축으로 정의
        # wandb.define_metric("epoch")

        # epoch 기준으로 그려질 metric 지정
        wandb.define_metric("train/epoch_*", step_metric="epoch")
        wandb.define_metric("validation/epoch_*", step_metric="epoch")

        # step 기준 metric (명시하지 않아도 되지만 가독성상 권장)
        wandb.define_metric("train/step_*")
        
    iter = 0

    ''' epoch '''
    for epoch in range(n_epoch):
        ''' train/step'''
        train_tracker = MetricTracker()

        train_loss_sum = 0
        epoch_train_loss_avg = 0
        train_correct_total = 0
        train_total = 0

        model.train()    # 모델 - 훈련모드

        for x, y in trainset_loader:
            # if iter < 400:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 0.01
            # elif iter == 400:
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 0.1 # 400 iter 이후 원래 LR 0.1 복구

            # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
            x_train = x.float().to(device)
            y_train = y.to(device)

            predicts = model(x_train)                   # 1. 예측
            train_loss = criterion(predicts, y_train)   # 2. 비용 함수            
            optimizer.zero_grad()                       # 3. gradient 초기화
            train_loss.backward()                       # 4. backward propagation
            optimizer.step()                            # 5. weight 업데이트

            train_loss_sum += train_loss.item()
            step_train_loss = train_loss.item()

            # accuracy
            with torch.no_grad():
                preds = predicts.argmax(dim=1)              # 각 샘플에 대해 가장 큰 logit을 가진 클래스 인덱스 반환
                correct = (preds == y_train).sum().item()   # 이번 배치에서 맞춘 갯수
                batch_size = y_train.size(0)
                step_train_acc = correct / batch_size

                train_correct_total += correct
                train_total += batch_size

            if run != None:
                run.log({
                    "train/step_loss": step_train_loss,
                    "train/step_acc": step_train_acc,
                })

            iter += 1

        epoch_train_loss_avg = train_loss_sum / len(trainset_loader)
        epoch_train_acc_avg = train_correct_total / train_total


        val_loss_sum = 0
        epoch_val_loss_avg = 0
        val_correct_total = 0
        val_total = 0

        model.eval()    # 모델 - 평가모드

        with no_grad():
            for x, y in valset_loader:
                x_train = x.float().to(device)    # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
                y_train = y.to(device)

                predicts = model(x_train)
                val_loss = criterion(predicts, y_train)

                val_loss_sum += val_loss.item()

                preds = predicts.argmax(dim=1)
                correct = (preds == y_train).sum().item()
                batch_size = y_train.size(0)

                val_correct_total += correct
                val_total += batch_size

            epoch_val_loss_avg = val_loss_sum / len(valset_loader)
            epoch_val_acc_avg = val_correct_total / val_total

        if run != None:
            run.log({
                "train/epoch_loss": epoch_train_loss_avg,
                "train/epoch_acc": epoch_train_acc_avg,
                "validation/epoch_loss": epoch_val_loss_avg,
                "validation/epoch_acc": epoch_val_acc_avg,
                "epoch": epoch+1,
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

        # 검증 손실 계산 후 모델 저장 로직 추가
        if epoch_val_loss_avg < best_val_loss:
            best_val_loss = epoch_val_loss_avg
            torch.save(model.state_dict(), f"{path}{time}/best_model_{time}.pt")
            print(f"--- Model saved at epoch {epoch+1} (Loss: {best_val_loss:.6f}) ---")
        
        # [추가] 에폭 종료 후 스케줄러 업데이트
        current_lr = optimizer.param_groups[0]['lr']

        if scheduler is not None:
            scheduler.step()
            # 현재 학습률 로깅 (선택 사항)
            current_lr = optimizer.param_groups[0]['lr']
            if run is not None:
                run.log({"common/learning_rate": current_lr, "epoch": epoch + 1})

        print(f'Epoch: {epoch+1:03d}/{n_epoch} | LR: {current_lr:.6f} | '
              f'Train Loss: {epoch_train_loss_avg:.4f} | Val Loss: {epoch_val_loss_avg:.4f} | '
              f'Val Acc: {epoch_val_acc_avg*100:.2f}%')

    if run != None:
        run.finish()

    return