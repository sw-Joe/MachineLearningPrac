from glob import glob

import numpy as np

# import pandas as pd
import torch.cuda
from cv2 import COLOR_BGR2RGB, INTER_LINEAR, cvtColor, imread, resize
from torch.utils.data import Dataset

# from shutil import copyfile



# PATH
PATH_ORIGINAL_CAT = "imgs/original/cat/"
PATH_ORIGINAL_DOG = "imgs/original/dog/"
PATH_TRAIN = "img/splited/train/"
PATH_TEST = "img/splited/test/"
IMG_FORMAT = "*.jpg"

TARGET_SIZE = 224    # 리사이징 목표 크기 / ?


"""  GPU 존재 확인 """
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" 데이터셋 전처리 """
class CustomDataset(Dataset): 
    """
    커스텀 데이터셋, 일종의 데이터 로더의 내부구현
    """

    def __init__(self, dir: str, label: int):
        """
        데이터셋의 전처리를 해주는 부분
        """
        self.dir = dir
        self.label = label
        self.img_list = glob(dir)


    def __len__(self):
        """
        데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
        """
        return len(self.img_list)


    def __getitem__(self, idx):
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


        """ 이미지 행렬 리사이징 함수 """
        h, w = matrix.shape[:2]
        scale = TARGET_SIZE / max(h, w)

        # 1. 비율 유지하며 리사이즈
        new_h, new_w = int(h * scale), int(w * scale)
        resized = resize(matrix, (new_w, new_h), interpolation=INTER_LINEAR)
        # resized = resized.astype(np.uint8)    # 2번 실행 과정에서 dtype오류 발생시

        # 2. 정사각형 캔버스 생성
        canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)

        # 3. 가운데 배치(padding)
        y = (TARGET_SIZE - new_h) // 2
        x = (TARGET_SIZE - new_w) // 2
        canvas[y:y+new_h, x:x+new_w] = resized

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()    # (C, H, W), 텐서로 변환
        tensor /= 255.    # Normalization
        tensor.to(device)

        return self.label, tensor