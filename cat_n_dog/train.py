from time import perf_counter

import torch.cuda


def count_time(func):
    def wrapper():
        t_start = perf_counter()
        f = func()
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
def train(model, optimizer, criterion, data_loader, n_epoch: int):
    """
    @param: model, optimizer, criterion, data_loader, n_epoch: int\n
    (1)예측 → (2)손실 계산 → (3)그래디언트 초기화 → (4)역전파 → (5)가중치 업데이트해보기
    """
    for epoch in range(n_epoch):
        avg_cost = 0
        total_batch = len(data_loader)

        # step
        for x, y in data_loader:
            x_train = x.float().to(device)    # 이미지 행렬을 선형 모델에 넣기 위한 형태인 1차원 벡터로 펼침(flatten)
            y_train = y.to(device)

            predicts = model(x_train)    # 1. 예측
            cost = criterion(predicts, y_train)  # 2. 비용 함수            
            optimizer.zero_grad()                # 3. gradient 초기화
            cost.backward()                      # 4. backward propagation
            optimizer.step()                     # 5. weight 업데이트

            avg_cost += cost / total_batch

        print('Epoch: {:02d}/{} | Cost: {:.6f}'.format(epoch+1, n_epoch, avg_cost))