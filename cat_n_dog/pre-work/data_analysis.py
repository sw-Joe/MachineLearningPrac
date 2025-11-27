from glob import glob

import numpy as np
import pandas as pd
from cv2 import COLOR_BGR2RGB, cvtColor, imread



# PATH
PATH_CAT = "../imgs/original/cat/"
PATH_DOG = "../imgs/original/dog/"
IMG_FORMAT = "*.jpg"


""" 데이터셋에 대한 분석 """
# 다음의 이유로 데이터 변환이 불가능한 이미지를 분별하여 데이터셋 정제
    # 이미지 포맷 등으로 읽을 수 없는 이미지
    # 불량 파일: 손상 또는 포맷 표준을 준수하지 않은 경우
'''
def verify_defective_img(img_matrix):
    special_format: bool = False
    defective: bool = False

    # JPEG, PNG, WEBP, GIF, BMP, TIFF
    # AVIF, HEIF ← HEIC은 HEIF로 표시됨
    img = Image.open(img_matrix)

    # 손상 이미지 분별

    print(img.format)    # DEBUG, HEIC indecated in HEIF
    return 
'''

def img_matrix_conversion(target_path: str) -> list:
    size_stats : list[list[int]] = [[], []]

    for file in glob(target_path):
        img_matrix = imread(file)  # BGR, HEIC(HEIF), AVIF 미지원

        # 읽을 수 없는 이미지를 처리에서 제외
        if img_matrix is None:
            # move(file, "imgs/_invalid/")    # 파일 이동
            # verify_defective_img(img_matrix)
            continue

        matrix = cvtColor(img_matrix, COLOR_BGR2RGB)    # [[[H, W, C], [], ...] ...]

        h, w, _ = matrix.shape
        size_stats[0].append(h)
        size_stats[1].append(w)

    return size_stats


if __name__ == "__main__":
    cat_img_stat = img_matrix_conversion(PATH_CAT+IMG_FORMAT)
    dog_img_stat = img_matrix_conversion(PATH_DOG+IMG_FORMAT)

    # 이미지의 높이와 너비에 대한 통계량을 얻어 리사이징 계획
    h_w_cat = np.array(cat_img_stat).T
    h_w_dog = np.array(dog_img_stat).T
    df_cat = pd.DataFrame(h_w_cat, columns=['Height_cat', 'Width_cat']).describe()
    df_dog = pd.DataFrame(h_w_dog, columns=['Height_dog', 'Width_dog']).describe()
    print(df_cat)
    print(df_dog)