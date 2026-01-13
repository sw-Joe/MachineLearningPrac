from glob import glob

import numpy as np
import torch.cuda
# from torchvision import transforms

import cv2 as cv
# from cv2 import COLOR_BGR2RGB, INTER_LINEAR, cvtColor, imread, resize
from torch.utils.data import Dataset



# transforms는 기본적으로 PIL 이미지 기반: 변환 필요
"""
transform = transforms.Compose([
    # transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
"""


"""  GPU 존재 확인 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" 데이터셋 전처리 """
class CustomDataset(Dataset): 

    def __init__(self, dir: str, label: int, target_resize: int = -1) -> None:
        self.dir: str = dir
        self.label: int = label
        self.img_list: list = glob(dir)
        self.target_size = target_resize


    def __len__(self):
        """데이터셋의 길이 반환"""
        return len(self.img_list)

    # 컬러이미지용
    def __getitem__(self, idx: int) -> tuple:
        """
        데이터셋에서 특정 1개의 샘플을 가져오는 함수
        단일 아이템 호출시 처리
        """
        img_matrix = cv.imread(self.img_list[idx])  # BGR, HEIC(HEIF), AVIF 미지원

        # 읽을 수 없는 이미지를 읽는 경우
        if img_matrix is None:
            print("can't read img", self.img_list[idx])

        matrix = cv.cvtColor(img_matrix, cv.COLOR_BGR2RGB)

        if self.target_size != -1:
            h, w = matrix.shape[:2]
            # 짧은 쪽이든 긴 쪽이든 기준에 맞춰 비율 유지 리사이즈
            scale = self.target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 리사이즈된 결과를 matrix 변수에 바로 할당
            matrix = cv.resize(matrix, (new_w, new_h), interpolation=cv.INTER_LINEAR)

        # numpy to torch tensor: (H, W, C) -> (C, H, W) C: channel
        tensor = torch.from_numpy(matrix).permute(2, 0, 1).float()
        tensor /= 255.  # Normalization
        tensor = tensor.to(device)

        return tensor, self.label

    # # 흑백이미지(Grayscale)용
    # def __getitem__(self, idx: int):
    #     img_path = self.img_list[idx]

    #     # 1. Grayscale 이미지 로드
    #     img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    #     if img is None:
    #         raise RuntimeError(f"이미지를 읽을 수 없습니다: {self.img_list[idx]}")

    #     # 2. float32 변환 및 정규화 [0,1]
    #     img = img.astype(np.float32) / 255.0

    #     # 3. (H, W) → (1, H, W)
    #     img = np.expand_dims(img, axis=0)

    #     # 4. Tensor 변환
    #     tensor = torch.from_numpy(img)

    #     return tensor, self.label
    

# 1. batch 데이터를 전부 불러와 병합 행렬로 만든다

# 2. 병합 행렬에서 데이터를 조회

# 3. 

class BinaryDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1) # (N, 32, 32, 3)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        # if self.transform:
        #     img = self.transform(img) # ToTensor, Normalize 등
        
        # 2. 기존 CustomDataset의 로직 적용: float32 변환 및 정규화
        # CIFAR-10 원본은 uint8이므로 연산을 위해 float32로 변환합니다.
        matrix = img.astype(np.float32)

        # 3. 텐서 변환 및 차원 변경 (H, W, C) -> (C, H, W)
        # 기존 코드의 .permute(2, 0, 1) 로직과 동일합니다.
        tensor = torch.from_numpy(matrix).permute(2, 0, 1)
        
        # 4. 정규화 (Normalization)
        tensor /= 255.0
        
        # 5. 장치 할당 (기존 코드의 .to(device) 반영)
        tensor = tensor.to(device)

        return tensor, label