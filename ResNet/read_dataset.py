import json
from glob import glob
from torch.utils.data import ConcatDataset
from ml_core.preprocessing import CustomDataset

class ImageNet100:
    def __init__(self, cfg, train_transform, val_transform):
        """
       의 데이터 로딩 로직을 클래스화함.
        - cfg: Hydra 설정 객체
        - train_transform: 학습용 이미지 증강
        - val_transform: 검증/테스트용 이미지 증강
        """
        self.cfg = cfg
        self.train_transform = train_transform
        self.val_transform = val_transform
        
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

    def _make_train_set(self):
        """4개의 train.X 디렉토리에서 데이터를 수집하여 결합"""
        custom_datasets_train = []
        train_dirs = [f"train.X{num}" for num in range(1, 5)]

        for directory in train_dirs:
            # dataset root dir/train_X{num}/*
            for folder in glob(self.cfg.dataset.dir + directory + "/*"):
                label_key = folder[-9:] # 파일명 마지막 9자리 추출
                if label_key in self.label_json:
                    int_label = self.label_to_idx[self.label_json[label_key]]
                    # CustomDataset은 (img, label, path) 3개를 반환함
                    custom_datasets_train.append(
                        CustomDataset(folder + self.ext, int_label, self.train_transform)
                    )
        
        return ConcatDataset(custom_datasets_train)

    def _make_test_set(self):
        """val.X 디렉토리에서 테스트 데이터를 수집"""
        custom_datasets_test = []
        val_list = glob(self.cfg.dataset.dir + "val.X/*")

        for folder in val_list:
            label_key = folder[-9:]
            if label_key in self.label_json:
                int_label = self.label_to_idx[self.label_json[label_key]]
                custom_datasets_test.append(
                    CustomDataset(folder + self.ext, int_label, self.val_transform)
                )
        
        return ConcatDataset(custom_datasets_test)

    def get_trainsets(self):
        """생성된 학습 데이터셋 반환"""
        self.train_set = self._make_train_set()
        return self.train_set
    
    def get_testsets(self):
        """생성된 테스트 데이터셋 반환"""
        return self.test_set
    
    def get_classes(self):
        """ 문자 라벨 """
        return self.classes