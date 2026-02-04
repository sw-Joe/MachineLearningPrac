from datetime import datetime
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import hydra
from omegaconf import DictConfig
import wandb
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from EfficientNet.read_dataset import ImageNet100
from EfficientNet.block import EfficientNet
# from ml_core.train import fit
from ml_core.train_module import Trainer
from ml_core.evaluation import ModelEvaluator
from ml_core.ema import EMA



@hydra.main(version_base=None, config_path="conf", config_name="b0_baseline")
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


    # 1. DistributedDataParallel 초기화
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        DEVICE = torch.device(f"cuda:{local_rank}")
    
    else:
        DEVICE = torch.device(cfg.device)
        
    print(f"torch.device: {DEVICE}")



    """ Wandb 기록 여부 플래그에 따른 초기화 """
    if cfg.wandb.enabled and (not dist.is_initialized() or dist.get_rank() == 0):
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
    # DEVICE = torch.device("cpu")
    torch.manual_seed(cfg.seed)                 # 난수 시드 고정, 단일 GPU
    g = torch.Generator().manual_seed(cfg.seed)
    # cuDNN: NVIDIA의 딥러닝 가속 라이브러리, Convolution 등을 빠르게 계산
    torch.backends.cudnn.deterministic = True   # cuDNN 라이브러리의 결정론적 알고리즘을 사용(성능 감소)
    torch.backends.cudnn.benchmark = False      # cuDNN의 자동 최적화 기능(auto-tuner)을 비활성화

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)        # 멀티 GPU 사용 시 권장
    #     DEVICE = torch.device(cfg.device)

    # print("torch.device:", DEVICE)


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
        transforms.TenCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = ImageNet100(cfg, train_transform, val_transform)

    test_set = dataset.get_testsets()
    classes = dataset.get_classes()

    ##########

    """ 모델 인스턴스 생성 """
    base_config = [
        [1, 16, 1, 1, 3], # Stage 2
        [6, 24, 2, 2, 3], # Stage 3
        [6, 40, 2, 2, 5], # Stage 4
        [6, 80, 3, 2, 3], # Stage 5
        [6, 112, 3, 1, 5], # Stage 6
        [6, 192, 4, 2, 5], # Stage 7
        [6, 320, 1, 1, 3], # Stage 8
    ]

    # 2. 모델 DDP 래핑
    model = EfficientNet(base_config, num_classes=cfg.model.num_classes).to(DEVICE)
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 3. DistributedSampler 적용
    if dist.is_initialized():
        """ customPackage.split을 이용한 데이터 분할(train, validation) """
        if FLAG:
            train_dataset = dataset.get_trainsets()
            # train(train + validation) 130000
            # test                        5000
            train_set, val_set = random_split(train_dataset, [105000, 25000], generator=g)

            train_sampler = DistributedSampler(train_set, shuffle=True)
            val_sampler = DistributedSampler(val_set, shuffle=True)

        test_sampler = DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None



    # model = EfficientNet(base_config, num_classes=100)  
    # if torch.cuda.device_count() > 1:
    #     # cuda:2 - master GPU, cuda:3 - sub
    #     model = nn.DataParallel(model, device_ids=[2, 3]).to(DEVICE)
    #     # model = nn.parallel.DistributedDataParallel(model, device_ids=[0, 1, 2, 3]).to(DEVICE)


    if FLAG:
        """ 옵티마이저, 비용함수 인스턴스 생성 """        
        # 설정값 - 논문 config
        base_lr = 0.256
        base_batch_size = 4096
        # current_batch_size = 128  # 사용자 환경에 맞게 수정

        # 1. Linear Scaling Rule 적용 - 사용자의 실행환경에 맞게 조정
        # global batch size -> cfg.train.batch_size(1024)
        scaled_lr = base_lr * (cfg.train.batch_size / base_batch_size)

        # Optimizer: RMSProp
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=scaled_lr,
            alpha=0.9,      # Decay (Rho)
            momentum=cfg.optimizer.momentum,
            eps=0.001,      # 수치 안정성을 위해 큰 값 사용
            weight_decay=cfg.optimizer.w_decay
        )

        # 2. 웜업 스케줄러 정의 (Linear Warmup)
        # 0 ~ warmup_epochs 동안 LR을 0에서 scaled_lr까지 선형 증가시킵니다.
        warmup_epochs = 5
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0        

        warmup_sch = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # 3. 메인 스케줄러 정의 (논문 사양: 2.4 에포크마다 0.97배)
        # 에포크 단위로 정밀하게 조절하기 위해 ExponentialLR을 사용하거나 
        # 설정값(cfg)에 따라 StepLR을 사용합니다.
        main_sch = StepLR(optimizer, 
                        step_size=cfg.scheduler.step_size, 
                        gamma=cfg.scheduler.gamma)

        # 4. SequentialLR 통합
        # milestones=[warmup_epochs]는 5에포크가 끝나는 시점에 스케줄러를 전환함을 의미합니다.
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_sch, main_sch], 
            milestones=[warmup_epochs]
        )

        ema = EMA(model, decay=0.999)

        criterion = nn.CrossEntropyLoss()    # 비용(손실)함수 객체 생성


    # global batch size 값 // GPU 개수 값을 적용
    batch_size_part=cfg.train.batch_size // (dist.get_world_size() if dist.is_initialized() else 1)

    if FLAG:
        """ 데이터 로더(데이터 탑재) """
        trainset_loader = DataLoader(train_set,
                                    batch_size=batch_size_part,
                                    sampler=train_sampler,
                                    num_workers=cfg.train.num_workers, 
                                    pin_memory=cfg.train.pin_memory,
                                    persistent_workers=cfg.train.persistent_worker, 
                                    prefetch_factor=cfg.train.prefetch_factor)
        valset_loader = DataLoader(val_set,
                                   batch_size=batch_size_part,
                                   sampler=val_sampler,
                                   num_workers=cfg.train.num_workers, 
                                   pin_memory=cfg.train.pin_memory,
                                   persistent_workers=cfg.train.persistent_worker, 
                                   prefetch_factor=cfg.train.prefetch_factor)


        """ 학습 """
        # 최적 모델을 저장
        fit(DEVICE, model, optimizer, scheduler, criterion, trainset_loader, valset_loader, 
            cfg.train.epochs, RUN, NOW, cfg.artifact.dir, ema)    # EMA instance added


    """ 모델 테스트 """
    ''' 테스트 데이터 로드 '''
    testset_loader = DataLoader(test_set,
                                batch_size=batch_size_part,
                                sampler=test_sampler,
                                num_workers=cfg.train.num_workers, 
                                pin_memory=cfg.train.pin_memory,
                                persistent_workers=cfg.train.persistent_worker, 
                                prefetch_factor=cfg.train.prefetch_factor)

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