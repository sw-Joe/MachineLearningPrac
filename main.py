from glob import glob

import cv2
import numpy as np
import pandas as pd
from PIL import Image


from torch.utils.data import DataLoader, Dataset
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import torchvision as tvio 



# PATH
PATH_CAT_IMG = "imgs/Cat/*.jpg"
PATH_DOG_IMG = "imgs/Dog/*.jpg"

'''
"""  GPU 존재 확인 """
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("torch.device:", device)
'''

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


size_stats : list[list[int]] = [[], []]

for file in glob(PATH_DOG_IMG):
    img_matrix = cv2.imread(file)  # BGR, HEIC(HEIF), AVIF 미지원

    ## 다음의 이유로 데이터 변환이 불가능한 이미지를 분별하여 데이터셋 정제
    # 이미지 포맷 등으로 읽을 수 없는 이미지
    # 불량 파일: 손상 또는 포맷 표준을 준수하지 않은 경우
    if img_matrix is None:
        # verify_defective_img(img_matrix)
        continue

    img = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1)    # 텐서로 변환

    # matrix = tvio.io.read_image(img)    # (C, H, W) uint8: (채널, 높이, 너비) 컴퓨터 비전 분야에서 사용되는 포맷
    matrix = tensor.permute(1, 2, 0)    # numpy에서 사용하기 위해 (H, W, C) 순서로 변경
    # matrix = matrix.permute(1, 2, 0).numpy()

    h, w, _ = matrix.shape
    size_stats[0].append(h)
    size_stats[1].append(w)

# 이미지의 높이와 너비에 대한 통계량을 얻어 리사이징 계획에 사용
h_w = np.array(size_stats).T
df = pd.DataFrame(h_w)
print(df.describe())

# # DEBUG
# n_img = len(glob(PATH_CAT_IMG))
# n_img_converted = len(size_stats)
# print(n_img, n_img_converted, n_img-n_img_converted)


'''
""" 데이터셋 전처리 """
# 커스텀 데이터셋
class CustomDataset(Dataset): 
    # 데이터셋의 전처리를 해주는 부분
    def __init__(self, dir):
        self.dir = dir
        self.img_list = glob(dir)


    # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
    def __len__(self):
        return len(self.img_list)


    # 데이터셋에서 특정 1개의 샘플을 가져오는 함수
    # 단일 아이템 호출시 처리
    def __getitem__(self, idx):
        img = tvio.read_image(self.img_list[idx])    # (C, H, W) uint8: (채널, 높이, 너비) 컴퓨터 비전 분야에서 사용되는 포맷

        matrix = img.permute(1, 2, 0).numpy()    # numpy에서 사용하기 위해 (H, W, C) 순서로 변경
        # h, w, _ = matrix.shape
        
        return matrix


def resize_with_padding(img, target_size=224):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)

    # 1) 비율 유지하며 리사이즈
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 2) 정사각형 캔버스 생성
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 3) 가운데 배치(padding)
    y = (target_size - new_h) // 2
    x = (target_size - new_w) // 2
    canvas[y:y+new_h, x:x+new_w] = resized

    return canvas



""" 모델 """
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.l


    def forward(self, xb):
        return self.model(xb)


"""
기본 훈련 루프
(1)예측 → (2)손실 계산 → (3)그래디언트 초기화
    → (4)역전파 → (5)가중치 업데이트해보기
"""
dataset = CustomDataset(PATH_CAT_IMG)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)


# 모델 생성(+ 모델을 GPU로 이동)
lm = Model().to('cuda')

# 옵티마이저 생성
optimizer = optim.SGD(lm.parameters(), lr=0.01)
# 손실함수 객체 생성
criterion = nn.MSELoss()


n_epoch = 1000
for epoch in range(n_epoch):
    # 예측()
    predicts = lm.forward(x_train)

    # 손실 계산
    cost = criterion(predicts, y_train)

    # gradient 초기화
    optimizer.zero_grad()

    cost.backward()   # backward propagation
    optimizer.step()    # weight 업데이트

    if epoch % 100 == 0:
        params = list(lm.parameters())
        print('Epoch: {:4d}/{} | Cost: {:.6f}'.format(
            epoch, n_epoch, cost.item()))
        

# Cat에 대한 학습, Dog에 대한 학습


""" 평가 """
# 두 개의 분류에 대한 특징을 통해 테스트셋을 분류
# 어떻게 됬던간에 두가지로만 나누면 된다
'''