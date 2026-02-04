import os
import random
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import torch.distributed as dist
import wandb



class EnvSetup:
    """딥러닝 실험 환경을 초기화하는 정적 메서드 모음"""

    @staticmethod
    def reproducibility(seed: int):
        """난수 시드 고정 및 cuDNN 결정론적 설정"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # [수정] random_split 등에서 사용할 전용 제너레이터 생성 및 반환
        g = torch.Generator()
        g.manual_seed(seed)
        
        # cuDNN 최적화 비활성화 및 결과 재현성 보장
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Setup] Reproducibility fixed with seed: {seed}")

        return g


    @staticmethod
    def distributed(backend: str="nccl"):
        """DDP 초기화 및 디바이스 정보를 반환"""
        is_dist = "LOCAL_RANK" in os.environ
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if is_dist:
            # NCCL 백엔드를 사용한 분산 프로세스 그룹 초기화
            dist.init_process_group(backend=backend)
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            rank = 0
            world_size = 1

        return {
            "is_dist": is_dist,
            "local_rank": local_rank,
            "rank": rank,
            "world_size": world_size,
            "device": device,
            "is_master": rank == 0
        }


    @staticmethod
    def logging(cfg, is_master: bool):
        """WandB 및 결과 저장 경로를 설정"""
        # 한국 시간 기준 타임스탬프 생성
        now = datetime.now(ZoneInfo("Asia/Seoul")).strftime('%y-%m-%d_%H-%M-%S')
        save_dir = Path(f"{cfg.artifact.dir}{now}")
        run = None

        if is_master:
            # 마스터 노드에서만 디렉토리 생성 및 WandB 초기화
            save_dir.mkdir(parents=True, exist_ok=True)
            if cfg.wandb.enabled:
                run = wandb.init(
                    project=cfg.wandb.project,
                    config=dict(cfg), # Hydra Config 전체를 WandB에 기록
                    tags=cfg.wandb.tags
                )
                print(f"[Setup] WandB Run initialized: {run.name}")
        
        return run, save_dir, now