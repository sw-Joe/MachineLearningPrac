import albumentations as A

from ml_core.transform import TransformBase



class MyUTKTransformConfig(TransformBase):
    """
    안면 데이터셋은 이목구비의 기하학적 구조가 중요하므로,
    이미지를 너무 심하게 뒤틀기보다는 현실적인 변형 위주로 구성
    """
    def __init__(self):
        # 공통으로 사용할 기본 전처리 (예: 리사이즈)
        self.base_resize = [A.Resize(224, 224)]

    @property
    def train_transform(self) -> A.Compose:
        # 일반 학습: 좌우 반전 및 미세한 회전/밝기 변화
        return self._build_pipeline(
            main_aug = self.base_resize + [
                A.HorizontalFlip(p=0.5), # 사람 얼굴의 대칭성 활용
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                                   rotate_limit=15, p=0.5), # 미세한 각도 조절
                A.RandomBrightnessContrast(p=0.2) # 조명 조건 대응
            ]
        )

    @property
    def strong_transform(self) -> A.Compose:
        # 소수 클래스(특정 인종/고령층): 데이터 다양성 극대화
        return self._build_pipeline(
            main_aug = self.base_resize + [
                A.HorizontalFlip(p=0.5),
                A.OneOf([
                    A.GaussNoise(p=1), # 저화질 이미지 대응
                    A.MotionBlur(p=1),
                ], p=0.3),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3)
            ],
            intermediate_aug = [
                A.CoarseDropout(max_holes=1, max_height=64, max_width=64, min_holes=1, p=0.3) # 얼굴 일부 가림 대응
            ]
        )

    @property
    def base_processor(self) -> A.Compose:
        # 검증용: 최소한의 규격화만 수행
        return self._build_pipeline(main_aug=self.base_resize)

    @property
    def tta_transform(self) -> A.Compose:
        # 테스트(TTA)용: 다양한 각도나 밝기 변화 적용
        return self._build_pipeline(
            # main_aug = self.base_resize + [A.RandomBrightnessContrast(p=1.0)]
            main_aug = self.base_resize
        )