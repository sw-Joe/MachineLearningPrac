from datetime import datetime
from zoneinfo import ZoneInfo

from evaluation import classification_eval
import hydra
from omegaconf import DictConfig
import wandb
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split


from ml_core.model import Model1GrayscaleMulticlass
from ml_core.preprocessing import CustomDataset
# from ml_core.split_data import DatasetSplit
from ml_core.train import fit
from ml_core.evaluation import (eval_confusion_matrix_multiclass, visualize_classification_results,
                                   visualize_mnist_results)



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
        "split": [5, 1, 1],
    }
    print(f"run name(time) : {NOW}")
    print(f"project : {PROJECT}")
    print(f"config : {CONFIG}")

    # True: 학습&평가 모드, False: 평가
    FLAG = True


    """ Wandb 기록 여부 플래그에 따른 초기화 """
    if cfg.wandb.enabled:
        RUN = wandb.init(
            project = PROJECT,
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
    # 동적 변수생성 방식(globals(), locals())
    # for i in (range(10)):
    #     globals()[f"dataset_{i}"] = CustomDataset(cfg.dataset.dir_train+"{i}/*.png", label=i)

    # train, val
    dataset_0 = CustomDataset(cfg.dataset.dir_train+"0/*.png", label=0)
    dataset_1 = CustomDataset(cfg.dataset.dir_train+"1/*.png", label=1)
    dataset_2 = CustomDataset(cfg.dataset.dir_train+"2/*.png", label=2)
    dataset_3 = CustomDataset(cfg.dataset.dir_train+"3/*.png", label=3)
    dataset_4 = CustomDataset(cfg.dataset.dir_train+"4/*.png", label=4)
    dataset_5 = CustomDataset(cfg.dataset.dir_train+"5/*.png", label=5)
    dataset_6 = CustomDataset(cfg.dataset.dir_train+"6/*.png", label=6)
    dataset_7 = CustomDataset(cfg.dataset.dir_train+"7/*.png", label=7)
    dataset_8 = CustomDataset(cfg.dataset.dir_train+"8/*.png", label=8)
    dataset_9 = CustomDataset(cfg.dataset.dir_train+"9/*.png", label=9)

    # test
    test_0 = CustomDataset(cfg.dataset.dir_test+"0/*.png", label=0)
    test_1 = CustomDataset(cfg.dataset.dir_test+"1/*.png", label=1)
    test_2 = CustomDataset(cfg.dataset.dir_test+"2/*.png", label=2)
    test_3 = CustomDataset(cfg.dataset.dir_test+"3/*.png", label=3)
    test_4 = CustomDataset(cfg.dataset.dir_test+"4/*.png", label=4)
    test_5 = CustomDataset(cfg.dataset.dir_test+"5/*.png", label=5)
    test_6 = CustomDataset(cfg.dataset.dir_test+"6/*.png", label=6)
    test_7 = CustomDataset(cfg.dataset.dir_test+"7/*.png", label=7)
    test_8 = CustomDataset(cfg.dataset.dir_test+"8/*.png", label=8)
    test_9 = CustomDataset(cfg.dataset.dir_test+"9/*.png", label=9)


    """ train, validation 분할 """
    proportion = [0.8, 0.2]
    train_0, val_0 = random_split(dataset_0, proportion, generator=g)
    train_1, val_1 = random_split(dataset_1, proportion, generator=g)
    train_2, val_2 = random_split(dataset_2, proportion, generator=g)
    train_3, val_3 = random_split(dataset_3, proportion, generator=g)
    train_4, val_4 = random_split(dataset_4, proportion, generator=g)
    train_5, val_5 = random_split(dataset_5, proportion, generator=g)
    train_6, val_6 = random_split(dataset_6, proportion, generator=g)
    train_7, val_7 = random_split(dataset_7, proportion, generator=g)
    train_8, val_8 = random_split(dataset_8, proportion, generator=g)
    train_9, val_9 = random_split(dataset_9, proportion, generator=g)

    train_set = train_0+train_1+train_2+train_3+train_4+train_5+train_6+train_7+train_8+train_9
    validation_set = val_0+val_1+val_2+val_3+val_4+val_5+val_6+val_7+val_8+val_9


    """ 모델, 옵티마이저, 비용함수 인스턴스 생성 """
    model = Model1GrayscaleMulticlass().to(DEVICE)    ### 모델 객체 생성(+ 모델을 GPU로 이동)
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

    test_set = test_0+test_1+test_2+test_3+test_4+test_5+test_6+test_7+test_8+test_9

    testset_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=False)

    if FLAG:
        pass
    else:
        NOW = "26-01-06_20-22-29"    # 테스트에 사용할 모델의 run_name(시간정보)를 명시

    path_model_status_saved = f"./best_model_{NOW}.pt"
    classes = [str(i) for i in range(10)]

    classification_eval(model, path_model_status_saved, testset_loader, classes)
    eval_confusion_matrix_multiclass(model, path_model_status_saved, testset_loader, classes, NOW)
    # visualize_classification_results(model, path_model_status_saved, testset_loader, classes, NOW)
    visualize_mnist_results(model, path_model_status_saved, testset_loader, NOW)


if __name__ == "__main__":
    main()