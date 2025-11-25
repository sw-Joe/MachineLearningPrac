import numpy as np
import torch
import torchvision.io as tvio



SRC = r"./img/test.jpeg"
RESULT = r"./img/result_rotate.jpeg"


# 이미지-행렬-이미지 변환(PNG)
def img_load(func):
    def wrapper(self, *args, **kwargs):

        img = tvio.read_image(self.src)    # (C, H, W) uint8: (채널, 높이, 너비) 컴퓨터 비전 분야에서 사용되는 포맷

        np_img = img.permute(1, 2, 0).numpy()    # numpy에서 사용하기 위해 (H, W, C) 순서로 변경
        # h, w, _ = np_img.shape

        processed = func(self, np_img, *args, **kwargs)    # 내부에서 실행될 함수

        out = torch.from_numpy(processed).permute(2, 0, 1).contiguous()    # 다시 (C, H, W)로 순서 변환
        # .contiguous() : 텐서의 메모리 배열을 연속적이도록 강제

        tvio.write_jpeg(out, self.result)    # 저장

        return processed
    return wrapper


class img_processing():
    def __init__(self, src, result):
        self.src = src
        self.result = result


    @img_load
    def rotate(self, img_matrix):
        # angle = np.clip(angle, 0, 359)
        # 점대칭 90도 회전 -> 행렬 전치(transpose) + 각 행을 뒤집기
        return img_matrix.transpose(1, 0, 2)[:, ::-1, :].copy()    # 전치 행렬


    @img_load
    def rl_inversion(self, img_matrix):
        # 가로축 반전(H, W, C)
        return img_matrix[:, ::-1, :].copy()    # 새로운 메모리 공간에 복사


    @img_load
    def td_inversion(self, img_matrix):
        # 세로축 반전(H, W, C)
        return img_matrix[::-1, :, :].copy()


    # def lightness(self, img_matrix, lightpoint):
    #     for h in range(img_matrix.shape[0]):
    #         for w in range(img_matrix.shape[1]):
    #             R = img[h, w, 0]
    #             G = img[h, w, 1]
    #             B = img[h, w, 2]
    #             for rgb in [R, G, B]:
    #                 rgb = np.clip(rgb + lightpoint, 0, 255)
    #     return img_matrix.copy()
    @img_load
    def lightness(self, img_matrix, lightpoint: int):
        # img_matrix : HWC
        result = img_matrix.astype(np.int32) + lightpoint
        result = np.clip(result, 0, 255)
        return result.astype(np.uint8)


img = img_processing(SRC, RESULT).rotate()