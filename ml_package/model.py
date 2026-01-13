import torch.nn as nn
import torch.nn.functional as F
from torch import flatten

""" 모델 정의 """
class Model1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)    # 입력 3채널, 출력 16채널
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.fc1 = nn.Linear(32 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)    # cat/dog → 2 classes

        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # 배치를 제외한 모든 차원을 평탄화(flatten)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = self.fc3(x)
        return x
    

class Model1GrayscaleMulticlass(nn.Module):    # Grayscale, Multiclass
    def __init__(self):
        super().__init__()

        # input size 28*28
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)   # (1,28,28) → (16,24,24)
        self.pool = nn.MaxPool2d(2, 2)                 # → (16,12,12)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # → (32,8,8)
                                                      # pool → (32,4,4)

        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)   # MNIST: 10 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class Model1ColorMulticlassMK2(nn.Module):    # RGB, 10 Classes
    def __init__(self):
        super().__init__()

        # 1. 입력 채널 수정: 1(흑백) -> 3(컬러)
        # Input size: (3, 32, 32)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)   # (3,32,32) → (16,28,28)
        self.pool = nn.MaxPool2d(2, 2)                 # (16,28,28) → (16,14,14)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)  # (16,14,14) → (32,10,10)
                                                       # MaxPool 적용 후 → (32,5,5)

        # 2. FC 레이어 입력 차원 계산 수정
        # 최종 특성맵 크기가 5x5 이므로 32 * 5 * 5 = 800
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)   # CIFAR-10: 10 classes

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))           # -> (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))           # -> (batch, 32, 5, 5)
        
        x = flatten(x, 1)                              # -> (batch, 800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                # 최종 출력 (10개 클래스 점수)
        return x
    

class Model1ColorMulticlassMK3(nn.Module):    # RGB, 10 Classes
    def __init__(self):
        super().__init__()

        # Input size: (3, 32, 32)
        # 커널 사이즈 3 적용 (padding=0 가정)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)   # (3,32,32) → (16,30,30)
        self.pool = nn.MaxPool2d(2, 2)                 # (16,30,30) → (16,15,15)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)  # (16,15,15) → (32,13,13)
                                                       # MaxPool 적용 후 → (32,6,6) ※ 소수점 버림

        # 2. FC 레이어 입력 차원 계산 수정
        # 최종 특성맵 크기가 6x6 이므로 32 * 6 * 6 = 1152
        self.fc1 = nn.Linear(32 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)   # CIFAR-10: 10 classes

    def forward(self, x):
        # x: (batch, 3, 32, 32)
        x = self.pool(F.relu(self.conv1(x)))           # -> (batch, 16, 15, 15)
        x = self.pool(F.relu(self.conv2(x)))           # -> (batch, 32, 6, 6)
        
        x = flatten(x, 1)                              # -> (batch, 1152)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                                
        return x

    

class Model2(nn.Module):
    # 기존 모델을 개선 또는 이진분류시 성능이 더뛰어난 비교 목적의 모델을 작성
    # 머신러닝 학습을 위한 간단한 모델

    def __init__(self):
        super().__init__()

        # Feature extractor
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 128 → 64 → 32
        # 최종 feature map: 32 × 32 × 32
        self.fc1 = nn.Linear(32 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 → 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 → 32
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),

            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x