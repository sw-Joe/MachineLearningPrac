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
        "epochs": 50,
        "batch_size": 32,
        "momentum": 0.9,
        "train, validation, test": "8:1:1",
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
    prev_val_loss: float = 1.0

    for epoch in range(n_epoch):
        avg_train_loss = 0
        avg_val_loss = 0
        total_batch = len(trainset_loader)

        # step(training set)
        for x, y in trainset_loader:
            x_train = x.float().to(device)    # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
            y_train = y.to(device)

            predicts = model(x_train)                   # 1. 예측
            train_loss = criterion(predicts, y_train)   # 2. 비용 함수            
            optimizer.zero_grad()                       # 3. gradient 초기화
            train_loss.backward()                       # 4. backward propagation
            optimizer.step()                            # 5. weight 업데이트

            avg_train_loss += train_loss / total_batch
            # Log metrics to wandb.
            run.log({"acc": avg_train_loss, "loss": train_loss})


        # step(validation set)
        with no_grad():
            for x, y in valset_loader:
                x_train = x.float().to(device)    # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
                y_train = y.to(device)

                predicts = model(x_train)
                val_loss = criterion(predicts, y_train)    ## **possibly unbound**

            # 과적합 탐지
            None if prev_val_loss >= val_loss else print("overfitting alert | " \
            "prev validation loss: {:.6f} | validation loss {:.6f}".format(prev_val_loss, val_loss))
        
            avg_val_loss += val_loss / total_batch    # validation 평균 loss
            prev_val_loss = val_loss

        print('Epoch: {:02d}/{} | training loss(avg): {:.6f} |'.format(epoch+1, n_epoch, avg_train_loss))

    # Finish the run and upload any remaining data.
    run.finish()
    return