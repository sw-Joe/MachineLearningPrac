from ml_package.metric import Metrics

from torch import load, max, no_grad
import torch.cuda
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


"""  GPU 존재 확인 """
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    # torch.cuda.manual_seed_all(SEED)    # 난수 제어
    DEVICE = torch.device("cuda")


""" 모델 평가 """
def model_test(model, model_status, test_loader, classes) -> None:
    """
    전체 데이터셋에 대한 평가
    """
    model.load_state_dict(load(model_status))
    model.eval()

    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)

    correct = 0
    total = 0

    # 각 분류(class)에 대한 예측값 계산을 위해 준비
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # 학습 중이 아니므로, 출력에 대한 변화도를 계산할 필요 x
    with no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            # 신경망에 이미지를 통과시켜 출력을 계산
            outputs = model(imgs)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택
            _, predicted = max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # 각 분류별 정확도(accuracy)를 출력
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    # 전체 정확도
    print(f'Accuracy of the network on the test_image_set: {100 * correct // total} %')


# def model_test_each_class(model, model_status, test_loader, classes) -> None:
#     """
#     어떤 것들을 더 잘 분류하고, 어떤 것들을 더 못했는지
#     """
#     model.load_state_dict(load(model_status))
#     model.eval()

#     dataiter = iter(test_loader)
#     imgs, labels = next(dataiter)

#     # 각 분류(class)에 대한 예측값 계산을 위해 준비
#     correct_pred = {classname: 0 for classname in classes}
#     total_pred = {classname: 0 for classname in classes}

#     with no_grad():
#         for data in test_loader:
#             imgs, labels = data
#             imgs = imgs.to(DEVICE)
#             labels = labels.to(DEVICE)
#             outputs = model(imgs)
#             _, predictions = max(outputs, 1)
#             # 각 분류별로 올바른 예측 수를 모읍니다
#             for label, prediction in zip(labels, predictions):
#                 if label == prediction:
#                     correct_pred[classes[label]] += 1
#                 total_pred[classes[label]] += 1


#     # 각 분류별 정확도(accuracy)를 출력
#     for classname, correct_count in correct_pred.items():
#         accuracy = 100 * float(correct_count) / total_pred[classname]
#         print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


def model_test_confusion_matrix(model, model_status_PATH, test_loader, classes, time) -> None:
    """
    이진 분류(cat vs dog)에 대한 confusion matrix 및 평가 지표 출력
    """
    model.load_state_dict(torch.load(model_status_PATH))
    model.eval()

    y_true = []
    y_pred = []

    with no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            _, predicted = max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Prediction")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix (Cat vs Dog)")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{time}.png")
    plt.close()

    # classification report
    print(classification_report(y_true, y_pred, target_names=classes))