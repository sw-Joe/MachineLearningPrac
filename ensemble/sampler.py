import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Sampler, WeightedRandomSampler



class DistributedWeightedSampler(Sampler):
    def __init__(self, dataset, weights, num_samples=None, replacement=True, seed=42):
        self.dataset = dataset
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement
        self.seed = seed
        
        # DDP 관련 설정
        if not dist.is_available() or not dist.is_initialized():
            self.num_replicas = 1
            self.rank = 0
        else:
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()

        # 각 GPU(replica)가 처리할 샘플 수
        self.num_samples_per_replica = int(np.ceil(self.num_samples / self.num_replicas))
        self.total_size = self.num_samples_per_replica * self.num_replicas
        self.epoch = 0

    def __iter__(self):
        # 매 에폭마다 다른 시드를 사용하여 셔플 효과 부여
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 전체 데이터셋에 대해 가중치 기반 샘플링 수행
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()

        # 현재 rank(GPU)에 해당하는 부분만 슬라이싱하여 할당
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples_per_replica

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_imbalance_tools(train_dataset):
    # 1. 제공해주신 클래스별 샘플 수 (알파벳 순서 권장)
    # akiec: 265, bcc: 414, bkl: 881, df: 87, mel: 895, nv: 5368, vasc: 110
    class_counts = [265, 414, 881, 87, 895, 5368, 110]
    
    # 2. 클래스별 가중치 계산 (1 / Count)
    # 샘플이 적을수록 큰 가중치를 가짐
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    
    # 3. 모든 훈련 샘플에 대한 개별 가중치 리스트 생성
    # ConcatDataset 내부를 순회하며 각 샘플의 라벨에 맞는 가중치 부여
    all_sample_weights = []
    for sub_ds in train_dataset.datasets:
        target_label = sub_ds.label
        weight = class_weights[target_label]
        all_sample_weights.extend([weight] * len(sub_ds))
    
    # 4. Sampler 생성 (교체 가능 추출 replacement=True 필수)
    sampler = WeightedRandomSampler(
        weights=all_sample_weights, 
        num_samples=len(all_sample_weights), 
        replacement=True
    )
    
    # 5. Loss Function용 가중치 (선택 사항이지만 병행 시 효과적)
    # 너무 극단적인 가중치를 방지하기 위해 정규화 수행
    loss_weights = class_weights / class_weights.sum() * len(class_counts)
    
    return sampler, loss_weights


# --- 실제 적용 예시 ---
# sampler, loss_weights = get_imbalance_tools(train_dataset)

# train_loader = DataLoader(
#     train_dataset, 
#     batch_size=cfg.train.batch_size, 
#     sampler=sampler, # shuffle=True는 제거해야 함
#     num_workers=cfg.train.num_workers
# )

# criterion = nn.CrossEntropyLoss(weight=loss_weights.to(device))