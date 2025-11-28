from glob import glob

import numpy as np
import torch.cuda
from torchvision import transforms

from cv2 import COLOR_BGR2RGB, INTER_LINEAR, cvtColor, imread, resize
from torch.utils.data import Dataset



TARGET_SIZE = 128    # 리사이징 목표 크기 / ?

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
    """커스텀 데이터셋, 일종의 데이터 로더의 내부구현"""

    def __init__(self, dir: str, label: int) -> None:
        """데이터셋의 전처리를 해주는 부분"""
        self.dir: str = dir
        self.label: int = label
        self.img_list: list = glob(dir)
        # self.transform = transform


    def __len__(self):
        """데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분"""
        return len(self.img_list)


    def __getitem__(self, idx: int) -> tuple:
        """
        데이터셋에서 특정 1개의 샘플을 가져오는 함수
        단일 아이템 호출시 처리
        """
        # img_name = self.img_list[idx]
        img_matrix = imread(self.img_list[idx])  # BGR, HEIC(HEIF), AVIF 미지원

        # 읽을 수 없는 이미지를 읽는 경우
        if img_matrix is None:
            print("can't read img", self.img_list[idx])

        matrix = cvtColor(img_matrix, COLOR_BGR2RGB)    # [[[H, W, C], [], ...] ...] 

        # transformer
        # transformed = self.transform(matrix)
        

        """ 이미지 행렬 리사이징 함수 """
        h, w = matrix.shape[:2]
        scale = TARGET_SIZE / max(h, w)

        # 1. 비율 유지하며 리사이즈
        new_h, new_w = int(h * scale), int(w * scale)
        resized = resize(matrix, (new_w, new_h), interpolation=INTER_LINEAR)
        # resized = resized.astype(np.uint8)    # 2번 실행 과정에서 dtype오류 발생시

        # 2. 정사각형 캔버스 생성
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.float32)

        # 3. 가운데 배치(padding)
        y = (TARGET_SIZE - new_h) // 2
        x = (TARGET_SIZE - new_w) // 2
        canvas[y:y+new_h, x:x+new_w] = resized

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()    # (C, H, W), 텐서로 변환
        tensor /= 255.    # Normalization
        tensor = tensor.to(device)

        return tensor, self.label