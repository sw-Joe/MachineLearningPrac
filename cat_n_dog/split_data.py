import torch
from parameter import BATCH_SIZE, IMG_FORMAT, PATH_ORIGINAL_CAT, PATH_ORIGINAL_DOG, SEED
from preprocessing import CustomDataset
from torch.utils.data import DataLoader, random_split

"""  GPU 존재 확인 """
device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)    # cuda 장치 전체의 전역 RNG(난수 생성기) 제어
    device = torch.device("cuda")


""" 데이터셋 객체 """
# 데이터에 대한 라벨링 및 처리기능
dataset_cat = CustomDataset(PATH_ORIGINAL_CAT + IMG_FORMAT, label=0)
dataset_dog = CustomDataset(PATH_ORIGINAL_DOG + IMG_FORMAT, label=1)


""" 데이터셋 분할 """

PROPORTION = 0.8
# cat, bog : 12475, 12470
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

torch.save(cat_test.indices, "cat_test_indices.pth")
torch.save(dog_test.indices, "dog_test_indices.pth")

# 고양이 + 강아지 데이터를 통합한 데이터셋
train_dataset = cat_train + dog_train

# 데이터 로더(데이터 탑재)
# batch_size가 작으면 GPU 사용 효과(병렬 연산의 장점)를 살리기 어려움
trainset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)