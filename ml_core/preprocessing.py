import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
import torchvision.transforms.functional as F



""" Dataset Preprocessing """
def rgb_matrix_transform(img_path):
    """
    컬러이미지(RGB)를 읽어 텐서로 변환 (Transform 미사용 시 대비)
    """
    try:
        img = Image.open(img_path).convert('RGB')
        # F.to_tensor는 [0, 255] -> [0, 1.0] 정규화와 (H,W,C)->(C,H,W) 변환을 동시에 수행합니다.
        return F.to_tensor(img)
    except Exception as e:
        print(f"이미지 로드 실패: {img_path} | 에러: {e}")
        return None


def grayscale_matrix_transform(img_path):
    """
    그레이스케일 이미지를 읽어 텐서로 변환
    """
    img = Image.open(img_path).convert('L')
    return F.to_tensor(img)


""" Customized Dataset 정의 """
class CustomDataset(Dataset): 
    def __init__(self, dir: str, label: int, transform=None) -> None:
        # glob(dir)의 dir은 이미 "path/*.JPEG" 형태여야 함
        self.img_list: list = sorted(glob(dir))
        self.label: int = label
        self.transform = transform

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx: int) -> tuple:
        img_path = self.img_list[idx]

        # 1. PIL로 이미지 열기 (OpenCV 대신)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 깨진 이미지 대비 예외 처리 (재귀적으로 다음 인덱스 호출)
            return self.__getitem__((idx + 1) % len(self))

        # 2. Transform 적용
        if self.transform:
            # PIL 이미지를 넘겨주면 RandomResizedCrop 등이 정상 작동합니다.
            return self.transform(img), self.label, img_path
        else:
            # Transform이 없으면 기본 텐서 변환만 수행
            return F.to_tensor(img), self.label, img_path


class BinaryDataset(Dataset):
    def __init__(self, labels, data, img_h: int=32, img_w: int=32, transform=None):
        self.labels = labels
        # CIFAR 형태의 평면 데이터를 (N, H, W, C)로 재구성
        self.data = data.reshape(-1, 3, img_h, img_w).transpose(0, 2, 3, 1) 
        self.transform = transform

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Numpy 배열 추출
        img_np = self.data[idx]
        label = self.labels[idx]

        # Numpy(uint8)를 PIL Image로 변환 (Transform 호환성 보장)
        img = Image.fromarray(img_np.astype('uint8'))

        if self.transform:
            return self.transform(img), label
        
        return F.to_tensor(img), label