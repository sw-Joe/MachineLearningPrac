import os
import glob
from pathlib import Path

import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# 사용자 정의 모듈 임포트
from dataset.UTKface.load_utkface import get_utk_splits
from .model import CNNRegression
from .transform import MyUTKTransformConfig
from ml_core.ema import EMA
from ml_core.engine.regression_engine import RegTrainer
from ml_core.evaluation.base import BaseEvaluator
from ml_core.setup import EnvSetup
from visualization import regression_scatterplot_visualization



def get_age_balanced_weights(dataset, num_bins=10):
    """
    데이터셋의 나이 분포를 분석하여 샘플별 가중치를 계산합니다.
    """
    # 1. 모든 샘플의 나이 데이터 추출
    # dataset[i]가 (image, age, ...) 형태라고 가정
    # ?? targets = [s[1] for s in train_set] # 모든 정답(나이) 추출
    all_ages = [int(dataset[i][1]) for i in range(len(dataset))]
    all_ages = np.array(all_ages)

    # 2. 빈도수(Frequency) 계산
    # 0세부터 116세까지의 실제 분포를 카운트합니다.
    counts = np.bincount(all_ages)
    
    # 3. 역수 기반 가중치 산출 (Smoothing 적용)
    # 데이터가 0개인 나이대의 ZeroDivision을 방지하기 위해 +1(Laplace Smoothing)을 하거나
    # log/sqrt를 취해 가중치가 너무 극단적으로 튀는 것을 방지합니다.
    weights = 1.0 / (np.sqrt(counts) + 1.0) # sqrt smoothing 추천
    
    # 4. 각 샘플에 가중치 매핑
    # 특정 샘플의 나이가 25세라면, weights[25] 값을 해당 샘플의 가중치로 할당합니다.
    sample_weights = [weights[age] for age in all_ages]
    
    return torch.DoubleTensor(sample_weights)


def get_smoothed_weights(dataset, alpha=0.3):
    all_ages = [int(dataset[i][1]) for i in range(len(dataset))]
    counts = np.bincount(all_ages)
    
    # alpha를 통해 가중치 격차 조절 (0에 가까울수록 균등해짐)
    weights = 1.0 / (np.power(counts, alpha) + 1e-6)
    
    # 특정 샘플이 너무 자주 뽑히지 않도록 가중치 상한선(Clipping) 설정
    # 예: 가장 흔한 샘플보다 최대 10배까지만 자주 뽑히도록 제한
    max_w = np.min(weights[weights > 0]) * 10
    weights = np.clip(weights, None, max_w)
    
    sample_weights = [weights[age] for age in all_ages]
    return torch.DoubleTensor(sample_weights)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    TRAIN_MODE = (cfg.mode == "train_n_eval")

    '''1. 환경 설정'''
    env = EnvSetup.distributed()
    _ = EnvSetup.reproducibility(cfg.seed)
    
    if TRAIN_MODE:
        run, save_dir, timestamp = EnvSetup.logging(cfg, env["is_master"])
    else:
        timestamp = cfg.get("eval_timestamp", "26-02-25_16-21-56")
        save_dir = Path(f"{cfg.artifact.dir}{timestamp}")

    '''2. 이미지 증강 및 데이터셋 설정'''
    utk_transform = MyUTKTransformConfig()
    
    # [수정] 나이 학습에만 집중하도록 데이터셋 분할 호출
    train_set, val_set, test_set = get_utk_splits(
        root_dir=cfg.dataset.dir,
        transform=utk_transform,
        test_size=cfg.dataset.split[1],
        val_size=cfg.dataset.split[2],
        seed=cfg.seed
    )

    '''3. 데이터 로더 구성'''
    loader_kwargs = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory
    }

    # 각 샘플에 대한 가중치 리스트 생성
    # sample_weights = get_age_balanced_weights(train_set)
    sample_weights = get_smoothed_weights(train_set, alpha=0.5)

    wr_splr = WeightedRandomSampler(weights=sample_weights, 
                                    num_samples=len(train_set),
                                    replacement=True)

    train_loader = DataLoader(train_set, sampler=wr_splr, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)

    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)


    '''4. 모델 및 손실 함수 설정'''
    # [수정] 회귀 전용 모델 및 MSE 손실 함수 설정
    model = CNNRegression().to(env["device"])
    
    # 회귀 손실 함수: MSE(Mean Squared Error) 또는 L1(MAE)
    criterion = nn.MSELoss().to(env["device"])     # qnem

    ema = EMA(model, decay=cfg.ema.decay)

    '''5. 학습 실행'''
    if TRAIN_MODE:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.base_lr)
        
        # Trainer 내부에서 labels['age']만 참조하도록 로직 확인 필요
        trainer = RegTrainer(
            device=env["device"], model=model, optimizer=optimizer, criterion=criterion, 
            config=cfg, ema=ema
        )
        trainer.fit(train_loader, val_loader, cfg.train.epochs, run, save_dir)

    ''' 평가 '''
    model_files = glob.glob(str(save_dir / "best_reg_model_ep*.pt"))
    best_model_path = model_files[0] if model_files else ""

    if best_model_path:
        true, pred = BaseEvaluator(model, device=env["device"])._pred(best_model_path, test_loader)

        regression_scatterplot_visualization(true, pred, save_dir, timestamp)

if __name__ == "__main__":
    main()