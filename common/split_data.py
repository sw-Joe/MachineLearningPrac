import torch
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset



""" 데이터셋 분할 """
# 다시 작성 : 데이터셋 객체와 분할 비율을 받아 분할한 데이터 로더를 반환
# cat, bog : 12475, 12470

class DatasetSplit:
    def __init__(self, dataset):
        self.dataset = dataset

    def t_v_t_split(self, proportion: list[float], save_title, save: bool = True) -> tuple[Subset, Subset, Subset]:
        """
        params: 
        분할 대상 데이터셋, 
        분할 비율 -> 데이터셋 수,
        작업:
        테스트셋의 인덱스를 export
        분할된 데이터셋을 튜플 형태로 export
        """
        if sum(proportion) > 1:
            print("proportion error: sum of proportion must not be bigger than 1")
            raise ValueError("proportion error")

        dataset_size = len(self.dataset)

        # def split_count(dataset_size):
        train_size = int(dataset_size * proportion[0])
        val_size   = int(dataset_size * proportion[1])
        if sum(proportion) < 1:
            test_size = int(dataset_size * proportion[2])
        else:
            test_size  = dataset_size - train_size - val_size    # 남은 것은 모두 test

        print("dataSet splited into :", train_size, val_size, test_size)
        train, val, test = random_split(self.dataset, [train_size, val_size, test_size])

        if save:
            torch.save(test.indices, f"{save_title}_test_indices.pth")    # 경로 수정 필요

        return train, val, test


    def load_trainset(self, model_name) -> Subset:
        test_idx  = torch.load(f"imgs/indices/{model_name}_test_indices.pth")    # 경로 수정 필요
        testset = Subset(self.dataset, test_idx)

        return testset