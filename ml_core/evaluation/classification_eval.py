import os

from sklearn.metrics import classification_report

from ml_core.evaluation.base import BaseEvaluator
from ml_core.metric.classification_metric import ClsTracker



""" 모델 평가 """
class ClassificationEvaluator(BaseEvaluator):
    """
    BaseEvaluator를 상속받아 구체적인 지표(Top-K, F1-Score)를 산출하는 클래스입니다.
    """
    def __init__(self, model, device, classes, time):
        # 상위 클래스의 생성자 호출을 통해 상태 초기화
        super().__init__(model, device)
        self.classes = classes
        self.time = time


    def add_top_k_error(self, model_path, loader, tag="Test"):
        """상위 클래스의 _prepare_model과 _inference_engine을 사용하여 Top-K 지표를 산출합니다."""
        with self._prepare_model(model_path):
            tracker = ClsTracker(topk=(1, 5), device=self.device) 
            for _, labels, outputs in self._inference_engine(loader):
                tracker.update(0, outputs, labels)

        tracker.synchronize()

        if self.rank == 0:    # 마스터 노드에서만 기록
            self.metrics_history.append({
                "type": "Top-K Error",
                "tag": tag,
                "timestamp": self.time,
                "top1_error": tracker.get_error_rate(1),
                "top5_error": tracker.get_error_rate(5),
                "accuracy": tracker.accuracy * 100
            })
            print(f"✅ {tag} Top-K 지표가 누적되었습니다.")


    def add_detailed_report(self, model_path, loader, tag="Detailed"):
        """상세 분류 리포트를 생성하고 히스토리에 추가합니다."""
        y_true, y_pred = self._pred(model_path, loader)

        # 2. 마스터 노드에서만 최종 리포트 생성
        if self.rank == 0:
            report_dict = classification_report(y_true, y_pred, target_names=self.classes, digits=3, output_dict=True)
            report_str = classification_report(y_true, y_pred, target_names=self.classes, digits=3)
        
            self.metrics_history.append({
                "type": "Classification Report",
                "tag": tag,
                "timestamp": self.time,
                "raw_str": report_str,
                "macro_f1": report_dict['macro avg']['f1-score']
            })
            
        print(f"✅ {tag} 상세 리포트가 누적되었습니다.")
        
        return y_true, y_pred


    def export(self, file_path):
        """누적된 지표들을 텍스트 파일로 추출합니다."""
        if not self.metrics_history:
            print("❌ No metric history Found.")
            return

        full_path = os.path.join(file_path, "eval_summary.txt")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write(f" EVALUATION REPORT ({self.time})\n")
            f.write("="*60 + "\n\n")

            for i, m in enumerate(self.metrics_history, 1):
                f.write(f"[{i}] {m['type']} - Tag: {m['tag']}\n")
                if m['type'] == "Top-K Error":
                    f.write(f" > Top-1 Error: {m['top1_error']:.2f}%\n")
                    f.write(f" > Top-5 Error: {m['top5_error']:.2f}%\n")
                    f.write(f" > Accuracy: {m['accuracy']:.2f}%\n")
                elif m['type'] == "Classification Report":
                    f.write(f" > Macro F1-Score: {m['macro_f1']:.4f}\n")
                    f.write(f" > Details:\n{m['raw_str']}\n")
                f.write("-" * 40 + "\n")
        
        ### 재검토
        if self.rank == 0:    # 마스터 노드에서만
            print(f"\n✅ 리포트 추출 완료 : {full_path}")