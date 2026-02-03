from contextlib import contextmanager
import json

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from torch import load, max, no_grad
import torch.cuda
from torch.amp.autocast_mode import autocast

from metric import MetricTracker



""" 모델 평가 """
class BaseEvaluator:
    """
    모든 평가 및 시각화 클래스의 모체가 되는 상위 클래스입니다.
    모델 상태 관리 및 추론 엔진 등 공통 핵심 로직을 포함합니다.
    """
    def __init__(self, model, device, classes, time):
        self.model = model
        self.device = device
        self.classes = classes
        self.time = time
        # 자식 클래스들이 지표를 누적할 수 있도록 초기화
        self.metrics_history = [] 

    @contextmanager
    def _prepare_model(self, model_path):
        """[private] 가중치 로드 및 평가 모드 전환을 담당하는 공통 컨텍스트 매니저입니다."""
        # map_location을 통해 device에 맞는 가중치 로드 보장
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        try:
            yield self.model
        finally:
            # 연산 후 GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def _inference_engine(self, loader):
        """
        [private] 중복되는 반복문, TenCrop 처리, AMP 설정을 관리하는 핵심 추론 제너레이터입니다.
        (img, label, path) 3개 인자 구조를 처리합니다.
        """
        for imgs, labels, _ in loader:
            # TenCrop 대응 로직: [Batch, 10, C, H, W] -> [Batch * 10, C, H, W]
            if len(imgs.shape) == 5:
                bs, n_crops, c, h, w = imgs.size()
                imgs = imgs.view(-1, c, h, w)
                
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            # 혼합 정밀도(AMP) 적용으로 연산 효율화
            with autocast(device_type=self.device.type, 
                          dtype=torch.float16 if self.device.type == 'cuda' else torch.bfloat16):
                outputs = self.model(imgs)
            
            # TenCrop 사용 시 10개의 결과값을 평균내어 최종 예측 산출
            if len(outputs) != len(labels):
                outputs = outputs.view(bs, n_crops, -1).mean(1)

            yield imgs, labels, outputs


class ModelEvaluator(BaseEvaluator):
    """
    BaseEvaluator를 상속받아 구체적인 지표(Top-K, F1-Score)를 산출하는 클래스입니다.
    """
    def __init__(self, model, device, classes, time):
        # 상위 클래스의 생성자 호출을 통해 상태 초기화
        super().__init__(model, device, classes, time)

    def add_top_k_error(self, model_path, loader, tag="Test"):
        """상위 클래스의 _prepare_model과 _inference_engine을 사용하여 Top-K 지표를 산출합니다."""
        with self._prepare_model(model_path):
            tracker = MetricTracker(topk=(1, 5)) 
            for _, labels, outputs in self._inference_engine(loader):
                tracker.update(0, outputs, labels)
        
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
        y_true, y_pred = [], []
        with self._prepare_model(model_path):
            for _, labels, outputs in self._inference_engine(loader):
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

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
        
        print(f"\n✅ 리포트 추출 완료 : {full_path}")