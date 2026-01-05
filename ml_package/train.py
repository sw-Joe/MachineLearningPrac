from time import perf_counter

# from config.log import log
from torch import no_grad
import torch.cuda
import wandb



"""  GPU 존재 확인 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_time(func):
    def wrapper(*args, **kwargs):
        t_start = perf_counter()
        f = func(*args, **kwargs)
        t_end = perf_counter()
        print("elapsed time: ", t_end-t_start)
        return f
    return wrapper


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



"""
기본 훈련 루프
"""
@count_time
def train(model, optimizer, criterion, trainset_loader, valset_loader, n_epoch: int, run, time) -> None:
    """
    @param: model, optimizer, criterion, trainset_loader, n_epoch: int\n
    (1)예측 → (2)손실 계산 → (3)그래디언트 초기화 → (4)역전파 → (5)가중치 업데이트해보기
    """

    early_stopping = EarlyStopping(
    patience=10,
    min_delta=1e-4
    )


    # Wandb 기록 여부를 체크
    if run != None:
        # epoch을 별도의 x축으로 정의
        # wandb.define_metric("epoch")

        # epoch 기준으로 그려질 metric 지정
        wandb.define_metric("train/epoch_*", step_metric="epoch")
        wandb.define_metric("validation/epoch_*", step_metric="epoch")

        # step 기준 metric (명시하지 않아도 되지만 가독성상 권장)
        wandb.define_metric("train/step_*")
        

    for epoch in range(n_epoch):
        train_loss_sum = 0
        epoch_train_loss_avg = 0
        train_correct_total = 0
        train_total = 0

        model.train()    # 모델 - 훈련모드

        ### 배치 로깅 추가 ###

        ''' step(mini-batch)'''
        for x, y in trainset_loader:
            x_train = x.float().to(device)    # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
            y_train = y.to(device)

            predicts = model(x_train)                   # 1. 예측
            train_loss = criterion(predicts, y_train)   # 2. 비용 함수            
            optimizer.zero_grad()                       # 3. gradient 초기화
            train_loss.backward()                       # 4. backward propagation
            optimizer.step()                            # 5. weight 업데이트

            train_loss_sum += train_loss.item()
            step_train_loss = train_loss.item()

            # 정확도
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


        epoch_train_loss_avg = train_loss_sum / len(trainset_loader)
        epoch_train_acc_avg = train_correct_total / train_total


        val_loss_sum = 0
        epoch_val_loss_avg = 0
        val_correct_total = 0
        val_total = 0

        model.eval()    # 모델 - 평가모드

        ''' step(validation set) '''
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


            is_best = early_stopping.step(epoch_val_loss_avg)

            if is_best:
                torch.save(model.state_dict(), f"best_model_{time}.pt")

            if early_stopping.should_stop:
                print(
                    f"Early stopping triggered at epoch {epoch+1} | "
                    f"best val loss: {early_stopping.best_loss:.6f}"
                )
                break

            # # 과적합 탐지
            # avg_val_loss += val_loss / total_batch    # validation 평균 loss
            # prev_val_loss = val_loss

        print('Epoch: {:02d}/{} | training loss: {:.6f} | validation loss: {:.6f}'
              .format(epoch+1, n_epoch, epoch_train_loss_avg, epoch_val_loss_avg))


    if run != None:
        run.finish()

    return