import glob
from pathlib import Path
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import hydra
from omegaconf import DictConfig

from ml_core.setup import EnvSetup
from ml_core.train_module import Trainer
from ml_core.ema import EMA
from ml_core.evaluation import ModelEvaluator
from ml_core.visualization import Visualize # 시각화 모듈 추가
from dataset.HAM10k.load_HAM10k import HAM10000
from ensemble.sampler import DistributedWeightedSampler, get_imbalance_tools
from ensemble.model import CNNEnsemble



def get_scheduler(optimizer, cfg):
    total_epochs = cfg.train.epochs
    warmup_epochs = 5 # 초기 안정화를 위한 구간
    
    # 1. Warmup: 0.2 * base_lr 에서 시작하여 1.0 * base_lr까지 선형 증가
    warmup_sch = LambdaLR(
        optimizer, 
        lr_lambda=lambda ep: (ep + 1) / warmup_epochs
    )
    
    # 2. Main: 나머지 에폭 동안 코사인 감쇄 적용 (일반화 성능 향상)
    main_sch = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs - warmup_epochs, 
        eta_min=1e-6 # 최저 학습률 설정
    )
    
    # 3. 통합: 5에폭(milestones) 지점에서 스케줄러 전환
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_sch, main_sch], 
        milestones=[warmup_epochs]
    )
    
    return scheduler


# 기존: criterion = nn.CrossEntropyLoss(weight=weights)
# 수정: 모델이 이미 log_softmax 형태를 반환하므로 NLLLoss 사용
# 단, Label Smoothing은 과적합 방지를 위해 필수입니다.

class LabelSmoothingNLLLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        # [수정] weight를 buffer로 등록하여 GPU 이동을 자동화합니다.
        if weight is not None and not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight, dtype=torch.float32)
        self.register_buffer('weight', weight)
        self.smoothing = smoothing
        
    def forward(self, log_probs, target):
        num_classes = log_probs.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        if self.weight is not None:
            # register_buffer 덕분에 self.weight가 target과 동일한 장치에 위치합니다.
            loss = loss * self.weight[target]
            
        return loss.mean()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # cv2.setNumThreads(0)

    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    
    # [추가] 모드 결정 플래그 (is_master와 독립적)
    TRAIN_MODE = (cfg.mode == "train_n_eval")


    '''1. 환경 설정 (Namespace 패턴 활용)'''
    env = EnvSetup.distributed()
    g = EnvSetup.reproducibility(cfg.seed)
    
    # 모드에 따른 로깅 및 경로 설정
    if TRAIN_MODE:
        run, save_dir, timestamp = EnvSetup.logging(cfg, env["is_master"])
    else:
        run = None
        # 평가 전용 모드일 때 사용할 기존 모델의 타임스탬프 (config에서 관리 권장)
        timestamp = cfg.get("eval_timestamp", "26-02-09_23-50-41")    # 26-02-04_23-55-04
        save_dir = Path(f"{cfg.artifact.dir}{timestamp}")


    '''2. 이미지 증강 정의 (Test용은 항상 필요)'''
    # ImageNet-1K 데이터셋의 통계치
    HAM10K_MEAN = (0.485, 0.456, 0.406)
    HAM10K_STD  = (0.229, 0.224, 0.225)    # 표준 편차

    def get_transforms(img_size, train=True):
        if train:
            return A.Compose([
                # 1. 기하학적 변형: 병변은 방향성이 없으므로 모든 각도 허용
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5), # 90도 단위 회전은 보간 왜곡이 적어 효과적임
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
                
                # 2. 색상 및 조명 변형: 촬영 환경(조명, 장비) 차이 극복
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
                ], p=0.4),
                
                # 3. 노이즈 및 질감: 모델이 미세한 질감에 집중하도록 유도
                A.GaussNoise(p=0.2),
                
                A.Normalize(HAM10K_MEAN, HAM10K_STD),
                ToTensorV2(),
            ])    # type: ignore
        else:
            return A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(HAM10K_MEAN, HAM10K_STD),
                ToTensorV2(),
            ])
        
    # 소수 클래스를 위한 더 강력한 변환 정의
    def get_strong_transforms(img_size):
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=45, p=0.7),
            # 기하학적 왜곡 추가 (의료 영상의 미세 특징 강화)
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=0.1, p=0.5),
            ], p=0.4),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(HAM10K_MEAN, HAM10K_STD),
            ToTensorV2(),
        ])  # type: ignore
        
    train_transform = get_transforms(224, train=True)
    train_strong_transform = get_strong_transforms(224)
    val_transform = get_transforms(224, train=False)


    '''3. 데이터셋 및 샘플러 설정'''
    # dataset 객체 생성 (transform은 내부에서 관리)
    dataset = HAM10000(cfg)
    # classes = dataset.get_classes()


    # 모드별 데이터 로딩 분기
    if TRAIN_MODE:
        train_set, val_set = dataset.get_split_datasets(train_transform, train_strong_transform, val_transform)
        
        # 1. 이전 단계에서 계산한 모든 샘플의 가중치 리스트 준비
        train_sampler, _ = get_imbalance_tools(train_set)
        all_sample_weights = train_sampler.weights

        # 2. 커스텀 DDP 가중치 샘플러 생성
        train_sampler = DistributedWeightedSampler(
            dataset=train_set,
            weights=all_sample_weights,
            replacement=True
        )

        # train_sampler = DistributedSampler(train_set, shuffle=True) if env["is_dist"] else None
        val_sampler = DistributedSampler(val_set, shuffle=False) if env["is_dist"] else None
    
    # test_set = dataset.get_testsets()
    # test_sampler = DistributedSampler(test_set, shuffle=False) if env["is_dist"] else None


    '''4. 데이터 로더 구성'''
    batch_size_part = cfg.train.batch_size // env["world_size"]
    loader_kwargs = {
        "batch_size": batch_size_part,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
        "persistent_workers": cfg.train.persistent_worker,
        "prefetch_factor": cfg.train.prefetch_factor
    }

    if TRAIN_MODE:
        train_loader = DataLoader(train_set, sampler=train_sampler, **loader_kwargs)
        val_loader = DataLoader(val_set, sampler=val_sampler, **loader_kwargs)
    
    # test_loader = DataLoader(test_set, sampler=test_sampler, **loader_kwargs)


    '''5. 모델 및 분산 설정 (항상 필요)'''
    # model = EfficientNet(base_config, num_classes=cfg.model.num_classes).to(env["device"])
    model = CNNEnsemble().to(env["device"])

    
    if env["is_dist"]:
        # [핵심 추가] 모든 GPU의 BN 통계를 동기화하여 수치 안정성 확보
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[env["local_rank"]], output_device=env["local_rank"])


    '''6 & 7. 학습 로직 (TRAIN_MODE일 때만 로딩)'''
    if TRAIN_MODE:
        # DDP 환경일 경우 model.module을 통해 실제 모델에 접근해야 합니다.
        base_model = model.module if env["is_dist"] else model
        
        optimizer = torch.optim.AdamW([
            # 1. ResNet-50: fc와 마지막 block 학습
            {"params": base_model.resnet.fc.parameters(), "lr": 1e-3},
            {"params": base_model.resnet.layer4.parameters(), "lr": 1e-4},
            
            # 2. EfficientNet-B4: classifier와 마지막 stage 학습
            {"params": base_model.effnet.classifier.parameters(), "lr": 1e-3},
            {"params": base_model.effnet.features[-1].parameters(), "lr": 1e-4},
            
            # 3. ViT-B/16: heads와 마지막 encoder layers 학습
            {"params": base_model.vit.heads.parameters(), "lr": 1e-3},
            {"params": base_model.vit.encoder.layers[-3:].parameters(), "lr": 1e-4},
        ], weight_decay=cfg.optimizer.w_decay)


        scheduler = get_scheduler(optimizer, cfg)


        ema = EMA(model, decay=cfg.ema.decay)


        # 이전에 계산한 counts = [265, 414, 881, 87, 895, 5368, 110] 기반
        counts = torch.tensor([265, 414, 881, 87, 895, 5368, 110])
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * 7.0
        # criterion = nn.CrossEntropyLoss(weight=weights.to(env["device"]),
        #                                 label_smoothing=0.1)
        criterion = LabelSmoothingNLLLoss(weight=weights, smoothing=0.1).to(env["device"])


        # 학습 실행
        trainer = Trainer(
            model=model, optimizer=optimizer, criterion=criterion, 
            scheduler=scheduler, device=env["device"], ema=ema, config=cfg
        )
        trainer.fit(train_loader, val_loader, cfg.train.epochs, run, save_dir)


    '''8. 최종 평가 (모든 GPU가 참여하여 병렬 처리)'''
    # 최적 모델 경로는 모든 GPU가 알고 있어야 함
    model_files = glob.glob(str(save_dir / "best_model_ep*.pt"))
    best_model_path = model_files[0] if model_files else ""

    if best_model_path:
        # 모든 Rank가 Evaluator 인스턴스 생성
        model_eval = ModelEvaluator(model, env["device"], dataset.get_classes(), timestamp)
        
        # 모든 Rank가 add_... 메서드에 진입 (내부에서 동기화 수행)
        model_eval.add_detailed_report(best_model_path, test_loader)
        model_eval.add_top_k_error(best_model_path, test_loader)
        
        # 모든 프로세스가 지표 계산을 마칠 때까지 대기
        if dist.is_initialized():
            dist.barrier()

        # 파일 추출만 마스터 노드에서 수행
        if env["is_master"]:
            model_eval.export(f"{save_dir}/")
            
            # [추가] 고확신 오답 분석 및 Grad-CAM 시각화
            json_files = sorted(glob.glob(str(save_dir / "misclassified_ep*.json")))
            if json_files:
                visualizer = Visualize(model, env["device"], dataset.get_classes(), timestamp)
                groups = visualizer.get_quartile_groups(json_files[-1])
                if groups:
                    visualizer.plot_confidence_grid(groups, save_dir)
                    # validation transform을 적용함에 유의
                    # visualizer.plot_gradcam_q4(groups, best_model_path, val_transform, save_dir)
        else:
            if env["is_master"]:
                print(f"[Warning] Best model not found in {save_dir}")



    # 안전한 종료
    try:
        # ... 기존 학습 및 평가 로직 ...
        model_eval.export(f"{save_dir}/")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group() # 분산 환경 자원 해제

if __name__ == "__main__":
    main()
    # import cv2
    # img_path = "dataset/HAM10k/competition_set/train/nv/ISIC_0024306.jpg"
    # img = cv2.imread(img_path)

    # print(type(img))