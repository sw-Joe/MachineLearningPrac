from datetime import datetime
from evaluation import classification_eval
import numpy as np
from pathlib import Path
import pickle
from zoneinfo import ZoneInfo

import hydra
from omegaconf import DictConfig
import wandb
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from ResNet.building_block import ResNet, BasicBlock
from ml_core.preprocessing import BinaryDataset
from ml_core.train import fit
from ml_core.evaluation import (eval_error, eval_confusion_matrix_multiclass, 
                                   visualize_cifar10_results, print_detailed_evaluation)



@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """ 로깅을 위한 시간변수 """
    kst_now = datetime.now(ZoneInfo("Asia/Seoul"))

    NOW = kst_now.strftime('%y-%m-%d_%H-%M-%S')
    PROJECT = cfg.wandb.project
    CONFIG = {
        "model_name": cfg.model.name,
        "dataset": cfg.wandb.metadata.dataset,
        "architecture": cfg.wandb.metadata.architecture,
        "optimizer": cfg.optimizer.name,
        "learning_rate": cfg.optimizer.lr,
        "momentum": cfg.optimizer.momentum,
        "epochs": cfg.train.epochs,
        "batch_size": cfg.train.batch_size,
        "split": cfg.dataset.split,
    }
    print(f"run name(time) : {NOW}")
    print(f"project : {PROJECT}")
    print(f"config : {CONFIG}")

    # artifact 저장 디렉토리 생성
    dir = Path(f"{cfg.artifact.dir}{NOW}")
    dir.mkdir(exist_ok=True)

    # True: 학습&평가 모드, False: 평가
    FLAG = True


    """ Wandb 기록 여부 플래그에 따른 초기화 """
    if cfg.wandb.enabled:
        RUN = wandb.init(
            # entity="sw-joe-kunkuk-glocal-university",
            project = PROJECT,
            # cfg의 모든 설정을 wandb config로 전달
            config = CONFIG,
            tags = cfg.wandb.tags
        )
    else:
        RUN = None


    """  GPU 존재 확인, 재현성 보장 """
    DEVICE = torch.device("cpu")
    torch.manual_seed(cfg.seed)                 # 난수 시드 고정, 단일 GPU
    g = torch.Generator().manual_seed(cfg.seed)
    # cuDNN: NVIDIA의 딥러닝 가속 라이브러리, Convolution 등을 빠르게 계산
    torch.backends.cudnn.deterministic = True   # cuDNN 라이브러리의 결정론적 알고리즘을 사용(성능 감소)
    torch.backends.cudnn.benchmark = False      # cuDNN의 자동 최적화 기능(auto-tuner)을 비활성화

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)        # 멀티 GPU 사용 시 권장
        DEVICE = torch.device(cfg.device)

    print("torch.device:", DEVICE)


    """ 이미지 증강(Augmentation) 정의 """
    # CIFAR-10 표준 정규화 값
    train_transform = transforms.Compose([
        transforms.ToPILImage(), # BinaryDataset 결과가 numpy/tensor일 경우 필요ㄴ
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4), # 논문 핵심 증강
        transforms.ToTensor(),
        # ResNet : 채널 단위 정규화 및 표준화
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # ResNet : 채널 단위 정규화 및 표준화
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


    """ train, test split """
    raw_train_data, raw_train_labels = load_cifar10_raw(cfg.dataset.dir, train=True)
    raw_test_data, raw_test_labels = load_cifar10_raw(cfg.dataset.dir, train=False)

    """ 커스텀 데이터셋 객체 선언 """
    # 데이터에 대한 라벨링 및 처리기능
    full_train_dataset = BinaryDataset(raw_train_data, raw_train_labels, train_transform)
    test_set = BinaryDataset(raw_test_data, raw_test_labels, test_transform)

    """ customPackage.split을 이용한 데이터 분할(train, validation) """
    if FLAG:
        train_set, val_set = random_split(full_train_dataset, [45000, 5000], generator=g)


    """ 모델 인스턴스 생성 """
    model = ResNet(BasicBlock, [200, 200, 200], num_classes=10).to(DEVICE)    ### 모델 객체 생성(+ 모델을 GPU로 이동)


    if FLAG:
        """ 옵티마이저, 비용함수 인스턴스 생성 """
        optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, 
                              momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.w_decay)    # 옵티마이저 생성: Stochastic Gradient Descent
        scheduler = MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma) # 에폭 기준 예시
        criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


        """ 데이터 로더(데이터 탑재) """
        # batch_size가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움
        trainset_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
        valset_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=True,
                                   num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)


        """ 학습 """
        # 최적 모델을 저장
        fit(model, optimizer, scheduler, criterion, trainset_loader, valset_loader, cfg.train.epochs, RUN, NOW, cfg.artifact.dir)


    """ 모델 테스트 """
    ''' 테스트 데이터 로드 '''
    testset_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)

    if FLAG:
        pass
    else:
        NOW = "26-01-22_02-11-37"    # 테스트에 사용할 모델의 run_name(시간정보)를 명시

    path_model_status_saved = f"./{cfg.artifact.dir}{NOW}/best_model_{NOW}.pt"
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # metric
    classification_eval(model, path_model_status_saved, testset_loader, classes)    # accuracy, precision, recall
    eval_error(model, path_model_status_saved, testset_loader)
    print_detailed_evaluation(model, path_model_status_saved, testset_loader, DEVICE)    # top-1, top-5 error rate
    # # visualization img
    eval_confusion_matrix_multiclass(model, path_model_status_saved, testset_loader, classes, NOW, cfg.artifact.dir)
    visualize_cifar10_results(model, path_model_status_saved, testset_loader, NOW, cfg.artifact.dir)


if __name__ == "__main__":
    main()