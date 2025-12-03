# import random

import hydra
from omegaconf import DictConfig
# from pathlib import Path
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ml_package.model import CNN
from ml_package.preprocessing import CustomDataset
from ml_package.split_data import DatasetSplit
from ml_package.train import train
from ml_package.test import model_test, model_test_each_class



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # if __name__ ==  '__main__':
    #     '''재현을 위한 시드 '''
    #     random.seed(SEED)

    """  GPU 존재 확인 """
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)    # 난수 제어
        DEVICE = torch.device(cfg.device)
    print("torch.device:", DEVICE)


    """ 커스텀 데이터셋 객체 선언 """
    # 데이터에 대한 라벨링 및 처리기능
    dataset_cat = CustomDataset(cfg.dataset.path_cat, label=0, target_resize=cfg.dataset.target_size)
    dataset_dog = CustomDataset(cfg.dataset.path_dog, label=1, target_resize=cfg.dataset.target_size)


    """ customPackage.split을 이용한 데이터 분할 """
    cat_train, cat_val, cat_test = DatasetSplit(dataset_cat).t_v_t_split(cfg.dataset.split, cfg.model_name)
    dog_train, dog_val, dog_test = DatasetSplit(dataset_dog).t_v_t_split(cfg.dataset.split, cfg.model_name)

    train_set = cat_train + dog_train
    validation_set = cat_val + dog_val


    """ 데이터 로더(데이터 탑재) """
    # batch_size가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움
    trainset_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True)
    valset_loader = DataLoader(validation_set, batch_size=cfg.train.batch_size, shuffle=True)


    """ 모델, 옵티마이저, 비용함수 인스턴스 생성 """ 
    cnn_model = CNN().to(DEVICE)    # 모델 생성(+ 모델을 GPU로 이동)
    optimizer = optim.SGD(cnn_model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum)    # 옵티마이저 생성: Stochastic Gradient Descent
    criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


    """ 학습 진행 """
    # train(cnn_model, optimizer, criterion, trainset_loader, valset_loader, TOTAL_EPOCH)


    """ 모델 상태 저장 """
    path_save = f"./{cfg.model_name}_cat_n_dog_CNN.pth"
    print("model: ", path_save)
    torch.save(cnn_model.state_dict(), path_save)


    """ 모델 테스트 """
    cat_test = DatasetSplit(dataset_cat).load_trainset(cfg.model_name)
    dog_test = DatasetSplit(dataset_dog).load_trainset(cfg.model_name)
    test_set = cat_test + dog_test

    testset_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=False)


    path_model_status_saved = f"./cat_n_dog_CNN_{cfg.train.batch_size}.pth"

    model_test(cnn_model, path_model_status_saved, testset_loader)
    classes = ['cat', 'dog']
    model_test_each_class(cnn_model, path_model_status_saved, testset_loader, classes)


if __name__ == "__main__":
    main()