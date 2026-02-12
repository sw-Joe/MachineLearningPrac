# Machine Learning Practice Repo.


## ref.
- https://tutorials.pytorch.kr/beginner/blitz/tensor_tutorial.html


## used dataset
### cat and dog classification with CNN
### MNIST
### CIFAR-10
### ImageNet-100
- https://www.kaggle.com/datasets/ambityga/imagenet100


## prac
### ImageNet
### ResNet
### EfficientNet


## 모델 로드 (Load)시 주의사항
학습할 때는 문제가 없지만, 나중에 저장된 모델을 불러올 때만 주의하시면 됩니다.

- DDP로 학습한 모델: 가중치 이름에 module.이라는 접두사가 붙어 있습니다.
- 단일 GPU로 학습한 모델: 접두사가 없습니다.


## 구조
### Dataset
#### Division
- training set
- validation set
- test set
### Model
### Optimizer
### Criterion - Loss(Cost) Function
### Learning Rate Scheduler
### Logging & Visualization