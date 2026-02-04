import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import hydra
from omegaconf import DictConfig

from ml_core.setup import EnvSetup
from ml_core.train_module import Trainer
from ml_core.ema import EMA
from ml_core.evaluation import ModelEvaluator
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
        timestamp = cfg.get("eval_timestamp", "26-00-00_00-00-00")
        save_dir = Path(f"{cfg.artifact.dir}{timestamp}")


    '''2. 이미지 증강 정의 (Test용은 항상 필요)'''
    test_transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.TenCrop(224), # Evaluation/Metric에서 처리
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
    dataset = ImageNet100(cfg, train_transform, test_transform)

    # 모드별 데이터 로딩 분기
    if TRAIN_MODE:
        train_dataset = dataset.get_trainsets()
        train_set, val_set = random_split(train_dataset, [105000, 25000], generator=g)
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
        model = DDP(model, device_ids=[env["local_rank"]], output_device=env["local_rank"])


    '''6 & 7. 학습 로직 (TRAIN_MODE일 때만 로딩)'''
    if TRAIN_MODE:
        # 최적화 도구 및 스케줄러 설정
        scaled_lr = 0.256 * (cfg.train.batch_size / 4096)
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=scaled_lr, alpha=0.9, momentum=cfg.optimizer.momentum,
            eps=0.001, weight_decay=cfg.optimizer.w_decay
        )
        warmup_sch = LambdaLR(optimizer, lr_lambda=lambda ep: (ep+1)/1- if ep < 10 else 1.0)
        main_sch = StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
        # main_sch = CosineAnnealingLR()
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sch, main_sch], milestones=[5])

        ema = EMA(model, decay=0.9999)
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
        
        # 파일 추출만 마스터 노드에서 수행
        if env["is_master"]:
            model_eval.export(f"{save_dir}/")
    else:
        if env["is_master"]:
            print(f"[Warning] Best model not found in {save_dir}")


if __name__ == "__main__":
    main()