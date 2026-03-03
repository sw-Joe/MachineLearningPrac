from abc import ABC, abstractmethod

import albumentations as A
from albumentations.pytorch import ToTensorV2



class TransformBase(ABC):
    """
    Albumentations 파이프라인의 표준 틀을 제공하는 추상 클래스입니다.
    """
    
    def _build_pipeline(self, main_aug=None, intermediate_aug=None):
        """
        정규화와 텐서 변환을 기본으로 하되, 
        그 사이(intermediate)나 앞(main)에 증강을 삽입하는 틀입니다.
        """
        pipeline = []
        
        # 1. 메인 증강 (회전, 플립, 크롭 등 일반적인 증강)
        if main_aug is not None:
            pipeline.extend(main_aug)
            
        # 2. 정규화 (필수: ImageNet 통계값 기준)
        pipeline.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))    # ImageNet-1K 산출값
        
        # 3. 정규화와 텐서 변환 사이의 추가 증강 (사용자 요청 '틀')
        if intermediate_aug is not None:
            pipeline.extend(intermediate_aug)
            
        # 4. 텐서 변환 (필수: PyTorch 연동)
        pipeline.append(ToTensorV2())
        
        return A.Compose(pipeline)
    


    @property
    @abstractmethod
    def train_transform(self): pass

    @property
    @abstractmethod
    def strong_transform(self): pass

    @property
    @abstractmethod
    def base_processor(self): pass

    @property
    @abstractmethod
    def tta_transform(self): pass