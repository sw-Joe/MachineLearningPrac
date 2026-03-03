from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .base import BaseMetricTracker



@dataclass
class RegTracker(BaseMetricTracker):
    sum_abs_error: torch.Tensor = field(init=False)
    sum_sq_error: torch.Tensor = field(init=False)

    def reset(self):
        super().reset()
        self.sum_abs_error = torch.tensor(0.0, device=self.device)
        self.sum_sq_error = torch.tensor(0.0, device=self.device)

    def update(self, loss_val: float, predicts, targets):
        batch_size = targets.size(0)
        predicts = predicts.view_as(targets)
        errors = predicts - targets
        
        self.total += batch_size
        self.loss_sum += loss_val * batch_size
        self.sum_abs_error += torch.abs(errors).sum()
        self.sum_sq_error += torch.pow(errors, 2).sum()

    def synchronize(self):
        super()._sync_common()
        if dist.is_initialized():
            dist.all_reduce(self.sum_abs_error, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.sum_sq_error, op=dist.ReduceOp.SUM)

    @property
    def mae(self): return (self.sum_abs_error / self.total).item()
    
    @property
    def rmse(self): return torch.sqrt(self.sum_sq_error / self.total).item()