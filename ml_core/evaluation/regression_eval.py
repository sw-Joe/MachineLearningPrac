from ml_core.evaluation.base import BaseEvaluator
from ml_core.metric.regression_metric import RegTracker



class RegressionEvaluator(BaseEvaluator):
    def add_regression_metrics(self, model_path, loader, tag="Test"):
        """테스트셋 MAE, RMSE 지표 산출"""
        with self._prepare_model(model_path):
            tracker = RegTracker(device=self.device)
            for _, labels, outputs in self._inference_engine(loader):
                # 회귀 평가 시 라벨 float 형변환 필수
                tracker.update(0, outputs, labels.float()) 

        tracker.synchronize()

        if self.rank == 0:
            self.metrics_history.append({
                "type": "Regression Metrics",
                "tag": tag,
                "mae": tracker.mae,
                "rmse": tracker.rmse,
                "avg_loss": tracker.avg_loss
            })
            print(f"✅ {tag} 회귀 지표 누적 완료 (MAE: {tracker.mae:.4f})")