import random

import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from common.model import CNN
from common.preprocessing import CustomDataset
from common.split_data import DatasetSplit
from common.train import train
from common.test import model_test, model_test_each_class
from cat_n_dog.config.parameter import BATCH_SIZE, IMG_FORMAT, PATH_ORIGINAL_CAT, PATH_ORIGINAL_DOG, SEED, TOTAL_EPOCH



# if __name__ ==  '__main__':
#     '''재현을 위한 시드 '''
#     random.seed(SEED)
RUN_NAME = "classification_1"

"""  GPU 존재 확인 """
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)    # 난수 제어
    DEVICE = torch.device("cuda")
print("torch.device:", DEVICE)


""" 커스텀 데이터셋 객체 선언 """
# 데이터에 대한 라벨링 및 처리기능
dataset_cat = CustomDataset(PATH_ORIGINAL_CAT + IMG_FORMAT, label=0)
dataset_dog = CustomDataset(PATH_ORIGINAL_DOG + IMG_FORMAT, label=1)


""" customPackage.split을 이용한 데이터 분할 """
PROPORTION = [0.7, 0.1, 0.2]
cat_train, cat_val, cat_test = DatasetSplit(dataset_cat).t_v_t_split(PROPORTION, RUN_NAME)
dog_train, dog_val, dog_test = DatasetSplit(dataset_dog).t_v_t_split(PROPORTION, RUN_NAME)

train_set = cat_train + dog_train
validation_set = cat_val + dog_val


""" 데이터 로더(데이터 탑재) """
# batch_size가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움
trainset_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
valset_loader = DataLoader(validation_set, batch_size=BATCH_SIZE, shuffle=True)


""" 모델, 옵티마이저, 비용함수 인스턴스 생성 """ 
cnn_model = CNN().to(DEVICE)    # 모델 생성(+ 모델을 GPU로 이동)
optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)    # 옵티마이저 생성: Stochastic Gradient Descent
criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


""" 학습 진행 """
# train(cnn_model, optimizer, criterion, trainset_loader, valset_loader, TOTAL_EPOCH)


""" 모델 상태 저장 """
path_save = f"./{RUN_NAME}_cat_n_dog_CNN.pth"
print("model: ", path_save)
torch.save(cnn_model.state_dict(), path_save)


""" 모델 테스트 """
cat_test = DatasetSplit(dataset_cat).load_trainset(RUN_NAME)
dog_test = DatasetSplit(dataset_dog).load_trainset(RUN_NAME)
test_set = cat_test + dog_test

testset_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


path_model_status_saved = f"./cat_n_dog_CNN_batch_32.pth"

model_test(cnn_model, path_model_status_saved, testset_loader)
classes = ['cat', 'dog']
model_test_each_class(cnn_model, path_model_status_saved, testset_loader, classes)