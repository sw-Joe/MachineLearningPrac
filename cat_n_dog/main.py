import random

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CNN
from parameter import PATH_SAVE, SEED, TOTAL_EPOCH
from split_data import trainset_loader
from train_loop import train

""" 학습을 위한 유저 입력"""
if __name__ ==  '__main__':
    '''재현을 위한 시드 '''
    random.seed(SEED)


    """  GPU 존재 확인 """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)    # 난수 제어
        device = torch.device("cuda")
    print("torch.device:", device)


    """ 모델, 옵티마이저, 비용함수 인스턴스 생성 """
    cnn_model = CNN().to(device)    # 모델 생성(+ 모델을 GPU로 이동)
    optimizer = optim.SGD(cnn_model.parameters(), lr=0.001, momentum=0.9)    # 옵티마이저 생성
    criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


    """ 학습 진행 """
    train(cnn_model, optimizer, criterion, trainset_loader, TOTAL_EPOCH)


    """ 모델 저장 """
    torch.save(cnn_model.state_dict(), PATH_SAVE)