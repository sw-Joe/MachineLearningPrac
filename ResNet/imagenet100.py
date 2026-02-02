from datetime import datetime
from pathlib import Path
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
from torchvision.transforms import InterpolationMode

from ResNet.building_block import ResNetImageNet, Bottleneck
from ResNet.read_dataset import ImageNet100
from ml_core.train import fit
from ml_core.evaluation import ModelEvaluator



@hydra.main(version_base=None, config_path="conf", config_name="bottleneck")
def main(cfg: DictConfig):
    # 데이터를 공유 메모리에 쌓지 않도록 강제
    # 성능저하 감수
    # torch.multiprocessing.set_sharing_strategy('file_system')

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

        # artifact 저장 디렉토리 생성
        dir = Path(f"{cfg.artifact.dir}{NOW}")
        dir.mkdir(exist_ok=True)
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

    # 1. 학습용 (Random Resized Crop 적용)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 검증/테스트용 (Center Crop 적용)
    val_transform = transforms.Compose([
        # 짧은 축을 256으로 리사이즈 (종횡비 유지)
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        # 중앙에서 224x224 크롭
        # transforms.CenterCrop(224),
        transforms.TenCrop(224),    # 10장의 PIL 이미지객체 반환
        # 10개의 크롭 이미지 각각에 대해 변환 적용 후 스택(Stack)
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    transforms.ToTensor()(crop)
                ) for crop in crops
            ]))
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    """ 커스텀 데이터셋 객체 선언 """
    dataset = ImageNet100(cfg, train_transform, val_transform)

    test_set = dataset.get_testsets()
    classes = dataset.get_classes()


    """ customPackage.split을 이용한 데이터 분할(train, validation) """
    if FLAG:
        train_dataset = dataset.get_trainsets()
        # train(train + validation) 130000
        # test                        5000
        train_set, val_set = random_split(train_dataset, [105000, 25000], generator=g)


    ##########


    """ 모델 인스턴스 생성 """
    model = ResNetImageNet(Bottleneck, [3, 4, 6, 3], num_classes=100)    ### 모델 객체 생성(+ 모델을 GPU로 이동)
    # model = ResNetImageNet(Bottleneck, [3, 4, 23, 3], num_classes=100)
        ### 모델 객체 생성(+ 모델을 GPU로 이동)

    if torch.cuda.device_count() > 1:
        # cuda:2 - master GPU, cuda:3 - sub
        model = nn.DataParallel(model, device_ids=[2, 3]).to(DEVICE)
        # model = nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3]).to(DEVICE)


    if FLAG:
        """ 옵티마이저, 비용함수 인스턴스 생성 """
        optimizer = optim.SGD(model.parameters(), lr=cfg.optimizer.lr, 
                              momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.w_decay)    # 옵티마이저 생성: Stochastic Gradient Descent

        scheduler = MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma) # 에폭 기준 예시

        # # 1. 웜업 스케줄러 (400 step까지 0.1배 적용)
        # warmup_sch = LambdaLR(optimizer, lr_lambda=lambda step: 0.1 if step < 400 else 1.0)
        # # 2. 메인 스케줄러 (30, 60, 90 에폭에서 감쇠)
        # main_sch = MultiStepLR(optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma)

        # # 3. 통합 (400 step 지점에서 메인으로 전환)
        # scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[400])

        criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


        """ 데이터 로더(데이터 탑재) """
        # batch_size가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움
        trainset_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                                    num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)
        valset_loader = DataLoader(val_set, batch_size=cfg.train.batch_size, shuffle=True,
                                   num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)


        """ 학습 """
        # 최적 모델을 저장
        fit(DEVICE, model, optimizer, scheduler, criterion, trainset_loader, valset_loader, cfg.train.epochs, RUN, NOW, cfg.artifact.dir)


    """ 모델 테스트 """
    ''' 테스트 데이터 로드 '''
    testset_loader = DataLoader(test_set, batch_size=cfg.train.batch_size, shuffle=False,
                                num_workers=cfg.train.num_workers, pin_memory=cfg.train.pin_memory)

    if FLAG:
        pass
    else:
        NOW = "26-00-00_00-00-00"    # 테스트에 사용할 모델의 run_name(시간정보)를 명시

    best_model = f"./{cfg.artifact.dir}{NOW}/best_model_{NOW}.pt"
    
    ''' metric '''
    model_eval = ModelEvaluator(model, DEVICE, classes, NOW)
    model_eval.add_detailed_report(best_model, testset_loader)
    model_eval.add_top_k_error(best_model, testset_loader)
    model_eval.export(f"./{cfg.artifact.dir}{NOW}/")


    ''' visualization '''
    # eval_confusion_matrix_multiclass(model, path_model_status_saved, testset_loader, classes, NOW, cfg.artifact.dir)
    # visualize_cifar10_results(model, path_model_status_saved, testset_loader, NOW, cfg.artifact.dir)


if __name__ == "__main__":
    main()