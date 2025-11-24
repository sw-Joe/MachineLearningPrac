import json

import torch
from torch.utils.data import DataLoader, random_split

from parameter import BATCH_SIZE, IMG_FORMAT, PATH_ORIGINAL_CAT, PATH_ORIGINAL_DOG, SEED
from preprocessing import CustomDataset

"""  GPU 존재 확인 """
device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)    # cuda 장치 전체의 전역 RNG(난수 생성기) 제어
    device = torch.device("cuda")


""" 데이터셋 분할 """
# cat/dog dataset
dataset_cat = CustomDataset(PATH_ORIGINAL_CAT + IMG_FORMAT, label=0)
dataset_dog = CustomDataset(PATH_ORIGINAL_DOG + IMG_FORMAT, label=1)

PROPORTION = 0.8
train_size_cat = int(len(dataset_cat) * PROPORTION)
train_size_dog = int(len(dataset_dog) * PROPORTION)
test_size_cat  = len(dataset_cat) - train_size_cat
test_size_dog  = len(dataset_dog) - train_size_dog

# 재현 가능한 split을 위한 generator 생성
# g = torch.Generator(device='cuda').manual_seed(SEED)

cat_train, cat_test = random_split(
    dataset_cat, [train_size_cat, test_size_cat]
)
dog_train, dog_test = random_split(
    dataset_dog, [train_size_dog, test_size_dog]
)


# train, test 분할을 저장하기 위한 json파일 생성
# 인덱스 저장 - indices : torch.utils.data.Subset
split_info = {
    "cat_train": cat_train.indices,
    "cat_test": cat_test.indices,
    "dog_train": dog_train.indices,
    "dog_test": dog_test.indices
}

with open("dataset_split.json", "w") as f:
    json.dump(split_info, f, indent=4)

print("Split saved → dataset_split.json")


# 고양이 + 강아지 데이터를 통합하여 사용
train_dataset = cat_train + dog_train
# test_dataset  = cat_test + dog_test

# 데이터 로더에 탑재
trainset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# testset_loader  = DataLoader(cat_test, batch_size=BATCH_SIZE)