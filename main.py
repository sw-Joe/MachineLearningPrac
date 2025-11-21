import random

# import numpy as np
# import pandas as pd
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import CNN

# from shutil import copyfile
from pre_precessing import CustomDataset
from train_loop import train

# PATH
PATH_ORIGINAL_CAT = "imgs/original/cat/"
PATH_ORIGINAL_DOG = "imgs/original/dog/"
PATH_TRAIN = "img/splited/train/"
PATH_TEST = "img/splited/test/"
IMG_FORMAT = "*.jpg"


""" 학습을 위한 유저 입력"""

'''재현을 위한 시드 '''
seed: int = int(input("무작위 생성을 위한 시드를 입력: "))
print(f"이 시드를 기억하세요: {seed}")
random.seed(seed)

''' 하이퍼파라미터 '''
total_epoch = 50
batch_size = 16


"""  GPU 존재 확인 """
device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
print("torch.device:", device)


""" 데이터셋 생성 """
dataset_cat = CustomDataset(PATH_ORIGINAL_CAT + IMG_FORMAT, label=0)    # cat: 0
dataset_dog = CustomDataset(PATH_ORIGINAL_DOG + IMG_FORMAT, label=1)    # dog: 1


""" 데이터 분할 """
proportion = 0.8    # 0과 1 사이의 값

train_size_cat = int(len(dataset_cat) * proportion)
test_size_cat  = len(dataset_cat) - train_size_cat
train_size_dog = int(len(dataset_dog) * proportion)
test_size_dog  = len(dataset_dog) - train_size_dog

cat_train, cat_test = random_split(dataset_cat, [train_size_cat, test_size_cat])
dog_train, dog_test = random_split(dataset_dog, [train_size_dog, test_size_dog])

# 고양이 + 강아지 데이터를 통합하여 사용
train_dataset = cat_train + dog_train
test_dataset  = cat_test  + dog_test

# 데이터 로더에 탑재
train_set_loader = DataLoader(cat_train, batch_size=batch_size, shuffle=True)

test_set_loader  = DataLoader(cat_test, batch_size=batch_size)


""" 모델, 옵티마이저, 비용함수 인스턴스 생성 """
cnn_model = CNN().to(device)    # 모델 생성(+ 모델을 GPU로 이동)
optimizer = optim.SGD(cnn_model.parameters(), lr=0.05)    # 옵티마이저 생성
criterion = nn.MSELoss()    # 비용(손실)함수 객체 생성


""" 학습 진행 """
train(cnn_model, optimizer, criterion, train_set_loader, total_epoch)