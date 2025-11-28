from time import perf_counter

# from config.log import log
from torch import no_grad
import torch.cuda
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="sw-joe-kunkuk-glocal-university",
    # Set the wandb project where this run will be logged.
    project="cat_n_dog_classification",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "Cat and Dog images",
        "optimizer": "Stochastic Gradient Descent(SGD)",
        "epochs": 50,
        "batch_size": 32,
        "momentum": 0.9,
        "train, validation, test": "7:1:2",
    },
)



def count_time(func):
    def wrapper(*args, **kwargs):
        t_start = perf_counter()
        f = func(*args, **kwargs)
        t_end = perf_counter()
        print("elapsed time: ", t_end-t_start)
        return f
    return wrapper


"""  GPU 존재 확인 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""
기본 훈련 루프
"""
@count_time
def train(model, optimizer, criterion, trainset_loader, valset_loader, n_epoch: int) -> None:
    """
    @param: model, optimizer, criterion, trainset_loader, n_epoch: int\n
    (1)예측 → (2)손실 계산 → (3)그래디언트 초기화 → (4)역전파 → (5)가중치 업데이트해보기
    """
    prev_val_loss: float = 1e9

    for epoch in range(n_epoch):
        train_loss_sum = 0
        epoch_train_loss_avg = 0
        train_correct_total = 0
        train_total = 0

        ''' step(mini-batch)(training set)'''
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

            # WandB 로깅
            run.log({
                "step_train_loss": step_train_loss,
                "step_train_acc": step_train_acc,
            })

        epoch_train_loss_avg = train_loss_sum / len(trainset_loader)
        epoch_train_acc = train_correct_total / train_total

        run.log({
            "epoch_train_loss": epoch_train_loss_avg,
            "epoch_train_acc": epoch_train_acc,
        })


        val_loss_sum = 0
        epoch_val_loss_avg = 0
        val_correct_total = 0
        val_total = 0

        ''' step(validation set) '''
        with no_grad():
            for x, y in valset_loader:
                x_train = x.float().to(device)    # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
                y_train = y.to(device)

                predicts = model(x_train)
                val_loss = criterion(predicts, y_train)    ## **possibly unbound**

                val_loss_sum += val_loss.item()
                step_val_loss = val_loss.item()

                preds = predicts.argmax(dim=1)
                correct = (preds == y_train).sum().item()
                batch_size = y_train.size(0)
                step_val_acc = correct / batch_size

                val_correct_total += correct
                val_total += batch_size

                run.log({
                    "step_val_loss": step_val_loss,
                    "step_val_acc": step_val_acc,
                })

            epoch_val_loss_avg = val_loss_sum / len(valset_loader)
            epoch_val_acc = val_correct_total / val_total

            run.log({
            "epoch_val_loss": epoch_val_loss_avg,
            "epoch_val_acc": epoch_val_acc,
            })

            # # 과적합 탐지
            # None if prev_val_loss >= step_val_loss else print("overfitting alert | " \
            # "prev validation loss: {:.6f} | validation loss {:.6f}".format(prev_val_loss, step_val_loss))
        
            # avg_val_loss += val_loss / total_batch    # validation 평균 loss
            # prev_val_loss = val_loss

        print('Epoch: {:02d}/{} | training loss: {:.6f} | validation loss: {:.6f}'
              .format(epoch+1, n_epoch, epoch_train_loss_avg, epoch_val_loss_avg))


    # Finish the run and upload any remaining data.
    run.finish()
    return