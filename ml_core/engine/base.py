import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast
from abc import ABC, abstractmethod



class BaseMetricTracker(ABC):
    @abstractmethod
    def reset(self): pass
    
    @abstractmethod
    def synchronize(self): pass
    
    @abstractmethod
    def update(self, loss_val, predicts, targets): pass


class BaseTrainer(ABC):
    def __init__(self, device, model, optimizer, criterion, scheduler=None, ema=None, config=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema
        self.config = config
        
        self.is_dist = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.is_master = (self.rank == 0)

        self.train_tracker: BaseMetricTracker | None = None
        self.val_tracker: BaseMetricTracker | None = None
        self._init_trackers()   # 자식 클래스에서 오버라이딩한 트래커 설정을 실행

    @abstractmethod
    def _init_trackers(self):
        """과제에 맞는 트래커(Cls/Reg)를 초기화해야 함"""
        pass

    # 추상메소드 인자 기입을 통한 시그니처 정교화
    @abstractmethod
    def _validate_epoch(self, loader):
        pass

    @abstractmethod
    def _handle_epoch_end(self, epoch, n_epochs, run_id, save_path, val_results):
        pass

    def fit(self, train_loader, val_loader, n_epochs, run_id, save_path):
        """전체 에포크 관리 루프"""
        for epoch in range(n_epochs):
            if self.is_dist and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            self._train_epoch(train_loader, run_id)
            val_results = self._validate_epoch(val_loader)

            self.train_tracker.synchronize()
            self.val_tracker.synchronize()

            if self.is_master:
                self._handle_epoch_end(epoch, n_epochs, run_id, save_path, val_results)
            
            self.train_tracker.reset()
            self.val_tracker.reset()
            torch.cuda.empty_cache()

    def _train_epoch(self, loader, run_id):
        """공통 순전파/역전파 로직"""
        self.model.train()

        # 오답 추론 이미지를 추출하려면 로더가 img, label, file_name 3개를 반환하도록
        # 프로그래밍해야 함
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True).bfloat16(), y.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predicts = self.model(x)
                loss = self.criterion(predicts, y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            if self.ema: self.ema.update()
            self.train_tracker.update(loss.item(), predicts, y)