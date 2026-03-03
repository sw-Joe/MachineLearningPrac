# import random
from datetime import datetime
from zoneinfo import ZoneInfo

from evaluation import classification_eval
import hydra
from omegaconf import DictConfig
import wandb
# from pathlib import Path
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# from ml_core.metric import Metrics
from ml_core.model import Model1, Model2
from ml_core.preprocessing import CustomDataset
from ml_core.split_data import DatasetSplit
from ml_core.train import fit
from ml_core.evaluation import eval_confusion_matrix_multiclass, visualize_classification_results



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


    """ 커스텀 데이터셋 객체 선언 """
    # 데이터에 대한 라벨링 및 처리기능
    dataset_cat = CustomDataset(cfg.dataset.path_cat, label=0, target_resize=cfg.dataset.target_size)
    dataset_dog = CustomDataset(cfg.dataset.path_dog, label=1, target_resize=cfg.dataset.target_size)

    """ customPackage.split을 이용한 데이터 분할 """
    if FLAG:
        cat_train, cat_val, cat_test = DatasetSplit(dataset_cat).t_v_t_split(
            cfg.dataset.split, cfg.model_name+"(cat)", g, save=True)
        dog_train, dog_val, dog_test = DatasetSplit(dataset_dog).t_v_t_split(
            cfg.dataset.split, cfg.model_name+"(dog)", g, save=True)
    else:
        cat_train, cat_val, cat_test = DatasetSplit(dataset_cat).t_v_t_split(
            cfg.dataset.split, cfg.model_name+"(cat)", g, False)
        dog_train, dog_val, dog_test = DatasetSplit(dataset_dog).t_v_t_split(
            cfg.dataset.split, cfg.model_name+"(dog)", g, False)

    train_set = cat_train + dog_train
    validation_set = cat_val + dog_val


    """ 모델, 옵티마이저, 비용함수 인스턴스 생성 """
    model = Model2().to(DEVICE)    ### 모델 객체 생성(+ 모델을 GPU로 이동)
    optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum)    # 옵티마이저 생성: Stochastic Gradient Descent
    criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


    if FLAG:
        """ 데이터 로더(데이터 탑재) """
        # batch_size가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움
        trainset_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True)
        valset_loader = DataLoader(validation_set, batch_size=cfg.train.batch_size, shuffle=True)


        """ 학습 """
        # 최적 모델을 저장
        fit(model, optimizer, criterion, trainset_loader, valset_loader, cfg.train.epochs, RUN, NOW)


    """ 모델 테스트 """
    ''' 테스트 데이터 로드 '''
    cat_test = DatasetSplit(dataset_cat).load_testset(cfg.model_name+"(cat)")
    dog_test = DatasetSplit(dataset_dog).load_testset(cfg.model_name+"(dog)")
    test_set = cat_test + dog_test

    testset_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=False)

    if FLAG:
        pass
    else:
        NOW = "26-01-06_20-22-29"    # 테스트에 사용할 모델의 run_name(시간정보)를 명시

    path_model_status_saved = f"./best_model_{NOW}.pt"
    classes = ['cat', 'dog']

    classification_eval(model, path_model_status_saved, testset_loader, classes)
    eval_confusion_matrix_multiclass(model, path_model_status_saved, testset_loader, classes, NOW)
    visualize_classification_results(model, path_model_status_saved, testset_loader, classes, NOW)


if __name__ == "__main__":
    main()