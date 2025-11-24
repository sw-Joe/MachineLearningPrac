import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

""" 모델 정의 """
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)    # 입력 3채널, 출력 16채널
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        # conv → pool 2번 후 출력 크기 계산
        # Input: 128×128
        # conv1 → 124×124
        # pool → 62×62
        # conv2 → 58×58
        # pool → 29×29

        self.fc1 = nn.Linear(32 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)    # cat/dog → 2 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x