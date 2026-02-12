from dataclasses import dataclass, field

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, classification_report)
import torch
import torch.distributed as dist


class MetricsLagacy:
    def __init__(self, test, predict):
        self.test = test
        self.predict = predict

    # 평가지표를 패키지가 아닌 직접 계산하도록 다시 작성
    def accuracy(self):
        return accuracy_score(self.test, self.predict)

    def precision(self):
        return precision_score(self.test, self.predict)

    def recall(self):
        return recall_score(self.test, self.predict)
        
    def f1(self):
        return f1_score(self.test, self.predict)

    def roc_auc(self):
        return roc_auc_score(self.test, self.predict)

    def confusion_matrix(self) -> None:
        sns.heatmap(confusion_matrix(self.test, self.predict), annot=True, fmt='d')
        plt.xlabel('Prediction')
        plt.ylabel('Real')
        plt.savefig('confusion_matrix.png')
        return
    
    # MAE, MSE, RMSE, MAPE


@dataclass
class MetricTracker:
    topk: tuple = (1, 5)
    device: torch.device = torch.device("cpu")
    # 내부 계산용 텐서 필드
    loss_sum: torch.Tensor = field(init=False)
    total: torch.Tensor = field(init=False)
    correct_counts: torch.Tensor = field(init=False)
    # loss_sum: float = 0.0
    # total: int = 0
    # correct_counts: dict = field(default_factory=dict)

    def __post_init__(self):
        self.reset()

    def reset(self):
        """지표 초기화 및 텐서 생성 (지정된 device 활용)"""
        self.loss_sum = torch.tensor(0.0, device=self.device)
        self.total = torch.tensor(0, device=self.device)
        self.correct_counts = torch.tensor([0.0] * len(self.topk), device=self.device)
        # self.loss_sum, self.total = 0.0, 0
        # self.correct_counts = {k: 0 for k in self.topk}
    
    @torch.no_grad()
    def update(self, loss_val: float, predicts, targets):
        """로컬 데이터를 누적 (이 시점엔 통신하지 않음)"""
        batch_size = targets.size(0)
        self.total += batch_size
        self.loss_sum += loss_val * batch_size    # loss_val은 float 가능
        
        # Top-k 계산 핵심 로직
        max_k = max(self.topk)
        _, pred = predicts.topk(max_k, 1, True, True)
        correct_matrix = pred.t().eq(targets.view(1, -1).expand_as(pred.t()))

        for i, k in enumerate(self.topk):
            # 수정: k(값)가 아닌 i(인덱스)를 사용
            # correct_matrix[:k]는 여전히 k(값)를 슬라이싱에 사용,
            # self.correct_counts[i]는 0번 혹은 1번 인덱스에 접근
            self.correct_counts[i] += correct_matrix[:k].reshape(-1).float().sum(0)

    def synchronize(self):
        """[핵심] DDP 환경인 경우 모든 GPU의 누적 지표를 합산 (Block 연산)"""
        if dist.is_initialized():
            # 모든 GPU의 값을 SUM하여 동기화
            dist.all_reduce(self.loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.total, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.correct_counts, op=dist.ReduceOp.SUM)

    @property
    def avg_loss(self):
        return (self.loss_sum / self.total).item() if self.total > 0 else 0.0

    @property
    def accuracy(self):
        """기존 정확도 지표 유지 (Top-1)"""
        return (self.correct_counts[0] / self.total).item() if self.total > 0 else 0.0

    def get_error_rate(self, k=1) -> float:
        if self.total == 0: return 0.0
        idx = self.topk.index(k)
        acc = (self.correct_counts[idx] / self.total).item()
        return (1.0 - acc) * 100
    

# [신규] F1-score 및 상세 리포트 전용 트래커
class ClassificationTracker:
    def __init__(self, classes=None):
        self.classes = classes
        self.reset()

    def reset(self):
        self.all_preds = []
        self.all_targets = []

    def update_batch(self, outputs, targets):
        """배치 예측값 수집 (GPU -> CPU 이동으로 메모리 보호)"""
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            self.all_preds.append(preds.cpu())
            self.all_targets.append(targets.cpu())

    def _gather_all(self):
        """DDP 환경에서 모든 GPU의 예측치를 마스터 노드로 수집"""
        preds = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)
        
        if dist.is_initialized():
            # 각 GPU의 결과를 리스트로 모음
            world_size = dist.get_world_size()
            gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
            gathered_targets = [torch.zeros_like(targets) for _ in range(world_size)]
            
            dist.all_gather(gathered_preds, preds)
            dist.all_gather(gathered_targets, targets)
            
            preds = torch.cat(gathered_preds)
            targets = torch.cat(gathered_targets)
            
        return preds.numpy(), targets.numpy()

    def get_f1_score(self, average='macro'):
        preds, targets = self._gather_all()
        return f1_score(targets, preds, average=average, zero_division=0)

    def get_report(self):
        preds, targets = self._gather_all()
        return classification_report(targets, preds, target_names=self.classes, zero_division=0)