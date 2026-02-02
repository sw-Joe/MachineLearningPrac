from dataclasses import dataclass, field

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)


class MetricsLagacy:
    def __init__(self, test, predict):
        self.test = test
        self.predict = predict

    # 평가지표를 패키지가 아닌 직접 계산하도록 다시 작성
    def _accuracy(self):
        return accuracy_score(self.test, self.predict)

    def _precision(self):
        return precision_score(self.test, self.predict)

    def _recall(self):
        return recall_score(self.test, self.predict)
        
    def _f1(self):
        return f1_score(self.test, self.predict)

    def _roc_auc(self):
        return roc_auc_score(self.test, self.predict)

    def _confusion_matrix(self) -> None:
        sns.heatmap(confusion_matrix(self.test, self.predict), annot=True, fmt='d')
        plt.xlabel('Prediction')
        plt.ylabel('Real')
        plt.savefig('confusion_matrix.png')
        return
    
    # MAE, MSE, RMSE, MAPE


@dataclass
class MetricTracker:
    topk: tuple = (1, 5)
    loss_sum: float = 0.0
    total: int = 0
    correct_counts: dict = field(default_factory=dict)

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.loss_sum, self.total = 0.0, 0
        self.correct_counts = {k: 0 for k in self.topk}
    
    @torch.no_grad()
    def update(self, loss_val: float, predicts, targets):
        batch_size = targets.size(0)
        self.total += batch_size
        self.loss_sum += loss_val * batch_size
        
        # Top-k 계산 핵심 로직
        max_k = max(self.topk)
        _, pred = predicts.topk(max_k, 1, True, True)
        correct_matrix = pred.t().eq(targets.view(1, -1).expand_as(pred.t()))

        for k in self.topk:
            self.correct_counts[k] += correct_matrix[:k].reshape(-1).float().sum(0).item()

    @property
    def avg_loss(self):
        return self.loss_sum / self.total if self.total > 0 else 0.0

    @property
    def accuracy(self):
        """기존 정확도 지표 유지 (Top-1)"""
        return self.correct_counts[1] / self.total if self.total > 0 else 0.0

    def get_error_rate(self, k=1):
        """ResNet 논문 재현용 에러율 (%)"""
        return (1.0 - (self.correct_counts[k] / self.total)) * 100 if self.total > 0 else 0.0