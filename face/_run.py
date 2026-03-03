import glob
from pathlib import Path
import os

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR # LinearLR 추가
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset.HAM10k.load_HAM10k import HAM10000
from ml_core.setup import EnvSetup
from engine.train import Trainer
from ml_core.ema import EMA
from ml_core.evaluation import ModelEvaluator
from ml_core.visualization import Visualizer    # 시각화 모듈 추가
from ensemble.sampler import DistributedWeightedSampler, get_imbalance_tools
from ensemble.model import CNNEnsembleByTimm    # timm 기반 pretrained 모델 로딩
from ensemble.augmentation import get_transforms, get_strong_transforms



def get_scheduler(optimizer, cfg):
    """[수정] LinearLR과 CosineAnnealingLR을 결합한 SequentialLR을 반환합니다."""
    total_epochs = cfg.train.epochs
    warmup_epochs = 5 # 초기 안정화를 위한 구간
    
    # 1. Warmup: 지정된 에폭 동안 학습률을 선형적으로 증가 (수치 안정성 확보)
    warmup_sch = LinearLR(
        optimizer, 
        start_factor=0.01, # 시작 시점의 비율 
        total_iters=warmup_epochs
    )
    # 2. Main: 나머지 에폭 동안 코사인 감쇄 적용 (일반화 성능 향상)
    main_sch = CosineAnnealingLR(
        optimizer, 
        T_max=total_epochs - warmup_epochs, 
        eta_min=1e-6 # 최저 학습률 설정
    )
    # 3. 통합: milestones 지점에서 스케줄러 전환
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_sch, main_sch], 
        milestones=[warmup_epochs]
    )
    
    return scheduler


# [추가] 레이어별 차등 학습률을 적용하기 위한 옵티마이저 빌더 함수
def get_optimizer(base_model, cfg):
    base_lr = cfg.optimizer.base_lr
    
    # 그룹별 파라미터 분리 및 개별 학습률 설정
    params = [
        # 1. 앙상블 가중치 및 모델별 Classifier (적극적 학습)
        {"params": [base_model.ensemble_weights], "lr": 1e-3},
        {"params": base_model.resnet.get_classifier().parameters(), "lr": base_lr},
        {"params": base_model.effnet.get_classifier().parameters(), "lr": base_lr},
        {"params": base_model.vit.get_classifier().parameters(), "lr": base_lr},
        
        # 2. ResNet-50: 상위 레이어 위주 미세 조정
        {"params": base_model.resnet.layer4.parameters(), "lr": base_lr * 0.1},
        {"params": base_model.resnet.layer3.parameters(), "lr": base_lr * 0.01},
        
        # 3. EfficientNet-B4: 마지막 블록들 위주
        {"params": base_model.effnet.blocks[5:].parameters(), "lr": base_lr * 0.1},
        {"params": base_model.effnet.blocks[3:5].parameters(), "lr": base_lr * 0.01},
        
        # 4. ViT: 마지막 인코더 레이어들 위주
        {"params": base_model.vit.blocks[-3:].parameters(), "lr": base_lr * 0.1},
        {"params": base_model.vit.blocks[-6:-3].parameters(), "lr": base_lr * 0.01},
    ]
    
    return torch.optim.AdamW(params, weight_decay=cfg.optimizer.w_decay)


class LabelSmoothingNLLLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
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
            loss = loss * self.weight[target]
            
        return loss.mean()


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    TRAIN_MODE = (cfg.mode == "train_n_eval")

    '''1. 환경 설정 (Namespace 패턴 활용)'''
    env = EnvSetup.distributed()
    g = EnvSetup.reproducibility(cfg.seed)
    
    if TRAIN_MODE:
        run, save_dir, timestamp = EnvSetup.logging(cfg, env["is_master"])
    else:
        run = None
        timestamp = cfg.get("eval_timestamp", "26-02-13_02-24-31")
        save_dir = Path(f"{cfg.artifact.dir}{timestamp}")


    '''2. 이미지 증강 정의 (Test용은 항상 필요)'''
    train_transform = get_transforms(224, train=True)
    train_strong_transform = get_strong_transforms(224)
    val_transform = get_transforms(224, train=False)


    '''3. 데이터셋 및 샘플러 설정'''
    dataset = HAM10000(cfg)

    if TRAIN_MODE:
        train_set, val_set = dataset.get_split_datasets(train_transform, train_strong_transform, val_transform)
        train_sampler_tool, _ = get_imbalance_tools(train_set)
        all_sample_weights = train_sampler_tool.weights

        train_sampler = DistributedWeightedSampler(
            dataset=train_set,
            weights=all_sample_weights,
            replacement=True
        )
        val_sampler = DistributedSampler(val_set, shuffle=False) if env["is_dist"] else None

    test_set = dataset.get_test_datasets(os.path.abspath("dataset/HAM10k/test_kaggle/"), val_transform)
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



    '''5. 모델 및 분산 설정'''
    # [수정] timm 기반 CNNEnsembleByTimm 로딩 및 Drop Path 적용
    model = CNNEnsembleByTimm(
        num_classes=cfg.model.num_classes,
        drop_path_rate=cfg.model.get('drop_path_rate', 0.35)
    ).to(env["device"])
    
    if env["is_dist"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[env["local_rank"]], output_device=env["local_rank"])


    '''6 & 7. 학습 로직'''
    if TRAIN_MODE:
        base_model = model.module if env["is_dist"] else model
        
        # [수정] 레이어별 차등 학습률 옵티마이저 호출
        optimizer = get_optimizer(base_model, cfg)

        scheduler = get_scheduler(optimizer, cfg)
        ema = EMA(model, decay=cfg.ema.decay)

        counts = torch.tensor([265, 414, 881, 87, 895, 5368, 110])
        weights = 1.0 / counts.float()
        weights = weights / weights.sum() * 7.0
        criterion = LabelSmoothingNLLLoss(weight=weights, smoothing=cfg.criterion.label_smoothing).to(env["device"])

        trainer = Trainer(
            device=env["device"], model=model, optimizer=optimizer, criterion=criterion, 
            scheduler=scheduler, ema=ema, config=cfg
        )
        trainer.fit(train_loader, val_loader, cfg.train.epochs, run, save_dir)


    '''8. 최종 평가'''
    model_files = glob.glob(str(save_dir / "best_model_ep*.pt"))
    best_model_path = model_files[0] if model_files else ""

    if best_model_path:
        model_eval = ModelEvaluator(model, env["device"], cfg.dataset.label, timestamp)
        # test_loader는 평가 전용 모드일 때 생성 로직이 필요할 수 있습니다.
        # 여기서는 TRAIN_MODE에서 파생된 val_loader를 예시로 사용하거나 기존 test_loader를 유지합니다.
        true, pred = model_eval.add_detailed_report(best_model_path, test_loader)
        
        if dist.is_initialized():
            dist.barrier()

        if env["is_master"]:
            model_eval.export(f"{save_dir}/")
            json_files = sorted(glob.glob(str(save_dir / "misclassified_ep*.json")))
            if json_files:
                visualizer = Visualizer(model, env["device"], cfg.dataset.label, timestamp)
                groups = visualizer.get_quartile_groups(json_files[-1])
                visualizer.confusion_matrix_visualization(true, pred, f"{save_dir}/")
                if groups:
                    visualizer.plot_confidence_grid(groups, save_dir)

    try:
        if env["is_master"]:
            model_eval.export(f"{save_dir}/")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()