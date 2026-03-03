from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.amp.autocast_mode import autocast



class BaseEvaluator:
    """
    모든 평가 및 시각화 클래스의 모체가 되는 상위 클래스
    모델 상태 관리 및 추론 엔진 등 공통 핵심 로직을 포함합니다.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
        # DDP 환경 변수 추가 정의
        self.is_dist = dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0
        self.world_size = dist.get_world_size() if self.is_dist else 1
        # 자식 클래스들의 지표 누적을 위한 리스트
        self.metrics_history = []

    @contextmanager
    def _prepare_model(self, model_path):
        """가중치 로드 및 평가 모드 전환 컨텍스트 매니저"""
        state_dict = torch.load(model_path, map_location='cpu')

        ## 현재 모델의 DDP wrapping 여부 확인
        if hasattr(self.model, 'module'):    # DDP 모델 : 내부 .module에 주입
            self.model.module.load_state_dict(state_dict)
        else:    # 일반 모델인 경우 바로 주입
            self.model.load_state_dict(state_dict)

        self.model.eval()
        try:
            yield self.model
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _inference_engine(self, loader):
        """[수정] 인덱스 기반 접근으로 (x, y) 또는 (x, y, path) 모두 대응"""

        # [private] 중복되는 반복문, TenCrop 처리, AMP 설정을 관리하는 핵심 추론 제너레이터
        # (img, label, _) 2개 인자를 사용, img에 대한 TTA 수행 후 산출된 output과 함께 반환
        # return img, label, output
        # """
        # for imgs, labels, _ in loader:
        #     # TenCrop 대응 로직: [Batch, 10, C, H, W] -> [Batch * 10, C, H, W]
        #     if len(imgs.shape) == 5:
        #         bs, n_crops, c, h, w = imgs.size()
        #         imgs =model_path, loader imgs.view(-1, c, h, w)

        for batch in loader:
            imgs, labels = batch[0], batch[1] # 데이터셋 구조에 유연하게 대응
            imgs = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast(device_type=self.device.type, dtype=torch.bfloat16):
                outputs = self.model(imgs)

        #     # TenCrop 사용 시 10개의 결과값을 평균내어 최종 예측 산출
        #     if len(outputs) != len(labels):
        #         outputs = outputs.view(bs, n_crops, -1).mean(1)
            
            yield imgs, labels, outputs

    # 기존 torch.max(outputs, 1)을 통한 분류에 대한 클래스 별 확률 중 최대값을
    #   pred 값으로 넘기는 것이 아닌 회귀 task에 맞는 raw outputs 값을 넘김
    def _pred(self, model_path, loader) -> tuple:
        y_true, y_pred = [], []
        with self._prepare_model(model_path):
            for _, labels, outputs in self._inference_engine(loader):
                # _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy().tolist())
                y_pred.extend(outputs.float().cpu().numpy().tolist())

        # DDP 환경에서 모든 Rank의 리스트 수집
        if self.is_dist:
            gathered_true = [None] * self.world_size
            gathered_pred = [None] * self.world_size
            dist.all_gather_object(gathered_true, y_true)
            dist.all_gather_object(gathered_pred, y_pred)
            
            if self.rank == 0:
                # 중첩 리스트 평탄화
                y_true = [i for sub in gathered_true for i in sub]
                y_pred = [i for sub in gathered_pred for i in sub]

        return y_true, y_pred