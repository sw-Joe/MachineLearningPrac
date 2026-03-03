import torch
import torch.distributed as dist
from dataclasses import dataclass, field

@dataclass
class BaseMetricTracker:
    device: torch.device = torch.device("cpu")
    loss_sum: torch.Tensor = field(init=False)
    total: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.reset()

    def reset(self):
        """공통 지표 초기화"""
        self.loss_sum = torch.tensor(0.0, device=self.device)
        self.total = torch.tensor(0, device=self.device)

    def _sync_common(self):
        """공통 텐서 동기화 (DDP)"""
        if dist.is_initialized():
            dist.all_reduce(self.loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)

    @property
    def avg_loss(self):
        return (self.loss_sum / self.total).item() if self.total > 0 else 0.0