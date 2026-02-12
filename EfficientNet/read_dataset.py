import json
import os
from glob import glob

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
import torchvision.transforms.functional as F

from ml_core.preprocessing import CustomDataset



""" Customized Dataset 정의 """
class CustomDatasetOverload(): 
    def __init__(self, dir_list: str, label: int, transform=None) -> None:
        # glob(dir)의 dir은 이미 "path/*.JPEG" 형태여야 함
        self.img_list: list = sorted(dir_list)
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



class ImageNet100:
    def __init__(self, cfg, test_transform):
        """
        ImageNet-100의 데이터 로딩 로직을 클래스화함.
        - cfg: Hydra 설정 객체
        - train_transform: 학습용 이미지 증강
        - test_transform: 테스트용 이미지 증강
        """
        self.cfg = cfg
        # self.train_transform = train_transform
        self.test_transform = test_transform
        
        self.ext = "/*.JPEG"
        self.label_json = self._load_labels()
        self.classes = sorted(list(set(self.label_json.values())))
        self.label_to_idx = {name: i for i, name in enumerate(self.classes)}
        
        # 데이터셋 객체 초기화
        # self.train_set = self._make_train_set()
        self.test_set = self._make_test_set()

    def _load_labels(self):
        """Labels.json 파일을 읽어 딕셔너리로 반환"""
        label_path = self.cfg.dataset.dir + "Labels.json"
        with open(label_path, "r") as f:
            return json.load(f)
        
    # read_dataset.py 내 ImageNet100 클래스 메서드 예시
    def get_split_datasets(self, train_transform, val_transform):
        train_datasets = []
        val_datasets = []

        for directory in [f"train.X{num}" for num in range(1, 5)]:
            base_dir = os.path.join(self.cfg.dataset.dir, directory)
            for folder in glob(base_dir + "/*"):
                label_key = os.path.basename(folder)
                if label_key in self.label_json:
                    label_idx = self.label_to_idx[self.label_json[label_key]]
                    
                    # 1. 해당 클래스의 모든 이미지 경로 수집
                    all_img_files = sorted(glob(folder + "/*.JPEG"))
                    
                    if not all_img_files: continue

                    # 2. 클래스 내부에서 분할 (Stratified Split 효과)
                    t_paths, v_paths = train_test_split(
                        all_img_files, test_size=0.19, 
                        random_state=self.cfg.seed, shuffle=True
                    )

                    # 3. [핵심] 분할된 '리스트'를 CustomDataset에 직접 전달
                    train_datasets.append(CustomDatasetOverload(t_paths, label_idx, train_transform))
                    val_datasets.append(CustomDatasetOverload(v_paths, label_idx, val_transform))

        return ConcatDataset(train_datasets), ConcatDataset(val_datasets)

    def _make_test_set(self):
        """val.X 디렉토리에서 테스트 데이터를 수집"""
        custom_datasets_test = []
        val_list = glob(self.cfg.dataset.dir + "val.X/*")

        for folder in val_list:
            label_key = folder[-9:]
            if label_key in self.label_json:
                int_label = self.label_to_idx[self.label_json[label_key]]
                custom_datasets_test.append(
                    CustomDataset(folder + self.ext, int_label, self.test_transform)
                )
        
        return ConcatDataset(custom_datasets_test)
    
    def get_testsets(self) -> ConcatDataset:
        """생성된 테스트 데이터셋 반환"""
        return self.test_set
    
    def get_classes(self):
        """ 문자 라벨 """
        return self.classes