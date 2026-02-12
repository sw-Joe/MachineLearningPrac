import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, StepLR ,CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode, Lambda
import hydra
from omegaconf import DictConfig

from ml_core.setup import EnvSetup
from ml_core.train_module import Trainer
from ml_core.ema import EMA
from ml_core.evaluation import ModelEvaluator
from ml_core.visualization import Visualize # 시각화 모듈 추가
from EfficientNet.read_dataset import ImageNet100
from EfficientNet.block import EfficientNet



@hydra.main(version_base=None, config_path="conf", config_name="b0_baseline")
def main(cfg: DictConfig):
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
        timestamp = cfg.get("eval_timestamp", "00-00-00_00-00-00")
        save_dir = Path(f"{cfg.artifact.dir}{timestamp}")


    '''2. 이미지 증강 정의 (Test용은 항상 필요)'''
    # 가장 올바른 Compose 구성 예시
    test_transform = transforms.Compose([
        transforms.Resize(256),
        # [Batch, 10, C, H, W]
        # 여기서 (img1, img2, ..., img10) 튜플 반환
        transforms.TenCrop(224),
        # Evaluation/Metric에서 처리하는 코드가 존재하나
        # 핵심 수정 부분: 튜플 내 각 이미지에 대해 ToTensor와 Normalize를 적용하고 스택함
        transforms.Lambda(lambda crops: torch.stack([
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(crop) for crop in crops
        ]))
    ])
    val_transform = transforms.Compose([
        # 1. 짧은 축을 256으로 리사이즈 (종횡비 유지)
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        # 2. 중앙에서 모델 입력 크기인 224x224만큼 크롭 (고정된 영역)
        transforms.CenterCrop(224),
        # 3. 텐서 변환 및 정규화 (학습과 동일한 파라미터)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if TRAIN_MODE:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = None



    '''3. 데이터셋 및 샘플러 설정'''
    # dataset 객체 생성 (transform은 내부에서 관리)
    dataset = ImageNet100(cfg, test_transform)

    # 모드별 데이터 로딩 분기
    if TRAIN_MODE:
        train_set, val_set = dataset.get_split_datasets(train_transform, val_transform)
        
        train_sampler = DistributedSampler(train_set, shuffle=True) if env["is_dist"] else None
        val_sampler = DistributedSampler(val_set, shuffle=False) if env["is_dist"] else None
    
    test_set = dataset.get_testsets()
    test_sampler = DistributedSampler(test_set, shuffle=False) if env["is_dist"] else None


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
    
    test_loader = DataLoader(test_set, sampler=test_sampler, **loader_kwargs)


    '''5. 모델 및 분산 설정 (항상 필요)'''
    base_config = [[1,16,1,1,3], 
                   [6,24,2,2,3], 
                   [6,40,2,2,5], 
                   [6,80,3,2,3], 
                   [6,112,3,1,5], 
                   [6,192,4,2,5], 
                   [6,320,1,1,3]]
    model = EfficientNet(base_config, num_classes=cfg.model.num_classes).to(env["device"])
    
    if env["is_dist"]:
        # [핵심 추가] 모든 GPU의 BN 통계를 동기화하여 수치 안정성 확보
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[env["local_rank"]], output_device=env["local_rank"])


    '''6 & 7. 학습 로직 (TRAIN_MODE일 때만 로딩)'''
    if TRAIN_MODE:
        # 최적화 도구 및 스케줄러 설정
        # scaled_lr = 0.256 * (cfg.train.batch_size / 4096)
        # 기존 0.256 또는 0.1에서 0.04로 대폭 하향
        scaled_lr = 0.04 * (cfg.train.batch_size / 256)
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=scaled_lr, alpha=0.9, momentum=cfg.optimizer.momentum,
            eps=0.01, weight_decay=cfg.optimizer.w_decay
        )    # original : eps=0.001
# 10에포크 동안 0.1, 0.2, ..., 1.0배로 증가
        warmup_epochs = 45
        warmup_sch = LambdaLR(optimizer, lr_lambda=lambda ep: (ep + 1) / warmup_epochs if ep < warmup_epochs else 1.0)
        main_sch = StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
        # main_sch = CosineAnnealingLR()
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[warmup_epochs])

        ema = EMA(model, decay=cfg.ema.decay)
        criterion = nn.CrossEntropyLoss()

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