import albumentations as A
from albumentations.pytorch import ToTensorV2



# Mean, Standard value for Normalization
IMAGENET1K_MEAN = (0.485, 0.456, 0.406)
IMAGENET1K_STD  = (0.229, 0.224, 0.225)

def get_transforms(img_size, train=True):
    if train:
        return A.Compose([
            # scale=(0.8, 1.0): 원본 이미지 면적의 80%에서 100% 사이만 크롭 영역으로 선택
            # p=1.0: 모든 학습 이미지에 대해 이 변환을 100% 확률로 적용
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
            
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
            
            # 다음 중 하나를 적용
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
            ]),

            A.GaussNoise(p=0.2),

            A.Normalize(IMAGENET1K_MEAN, IMAGENET1K_STD),
            ToTensorV2(),
        ]) # type: ignore
    else:
        return A.Compose([
            A.Resize(img_size, img_size),

            A.Normalize(IMAGENET1K_MEAN, IMAGENET1K_STD),
            ToTensorV2(),
        ])
    

def get_strong_transforms(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), p=1.0),
        
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=45, p=0.7),
        
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=0.1, p=0.5),
        ], p=0.4),

        A.RandomBrightnessContrast(p=0.5),

        A.Normalize(IMAGENET1K_MEAN, IMAGENET1K_STD),
        ToTensorV2(),
    ]) # type: ignore


def get_strong_transforms2(img_size):
    return A.Compose([
        # [핵심] 8% 영역까지 과감하게 크롭하여 모델의 관찰력 극대화
        A.RandomResizedCrop(
            size=(img_size, img_size), 
            scale=(0.08, 1.0), 
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # 최신 ColorJitter 적용 (v1.4+ 권장)
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
        
        # [최신 API] CoarseDropout: 정보 손실을 통한 규제 강화
        A.CoarseDropout(
            num_holes_range=(4, 10), 
            hole_height_range=(0.05, 0.15), 
            hole_width_range=(0.05, 0.15), 
            fill='random_uniform', 
            p=0.5
        ),
        
        A.Normalize(IMAGENET1K_MEAN, IMAGENET1K_STD),
        ToTensorV2(),
    ])


# 병행했을 때 효과가 극대화되는 TTA 전용 ModelEvaluator 코드 추가
# 10-Crops TTA(Test-Time Augmentation)
def get_five_crop_transforms(img_h, img_w, crop_size):
    """
    원본 이미지 크기(img_h, img_w)와 자를 크기(crop_size)를 기준으로
    5개 위치의 Crop 변환 리스트를 반환합니다.
    """
    h, w = crop_size, crop_size
    
    return [
        A.Crop(x_min=0, y_min=0, x_max=w, y_max=h, p=1),               # 1. 좌상(Top-Left)
        A.Crop(x_min=img_w-w, y_min=0, x_max=img_w, y_max=h, p=1),     # 2. 우상(Top-Right)
        A.Crop(x_min=0, y_min=img_h-h, x_max=w, y_max=img_h, p=1),     # 3. 좌하(Bottom-Left)
        A.Crop(x_min=img_w-w, y_min=img_h-h, x_max=img_w, y_max=img_h, p=1), # 4. 우하(Bottom-Right)
        A.CenterCrop(height=h, width=w, p=1)                           # 5. 중앙(Center)
    ]