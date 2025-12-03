from torch import load, max, no_grad
import torch.cuda



"""  GPU 존재 확인 """
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    # torch.cuda.manual_seed_all(SEED)    # 난수 제어
    DEVICE = torch.device("cuda")


# PARAMETER를 json파일로 저장하는 경우 불러오기
# with open("dataset_split.json", "r") as f:
#     test_list_idxs = json.load(f)    # 파일명 목록의 인덱스에 해당하는 기록들의 모음


""" 모델 평가 """
def model_test(model, model_status_PATH, test_loader) -> None:
    """
    전체 데이터셋에 대한 평가
    """
    model.load_state_dict(load(model_status_PATH))

    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)

    correct = 0
    total = 0

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

    print(f'Accuracy of the network on the test_image_set: {100 * correct // total} %')


def model_test_each_class(model, model_PATH, test_loader, classes) -> None:
    """
    어떤 것들을 더 잘 분류하고, 어떤 것들을 더 못했는지
    """
    model.load_state_dict(load(model_PATH))

    dataiter = iter(test_loader)
    imgs, labels = next(dataiter)

    # 각 분류(class)에 대한 예측값 계산을 위해 준비
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(imgs)
            _, predictions = max(outputs, 1)
            # 각 분류별로 올바른 예측 수를 모읍니다
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # 각 분류별 정확도(accuracy)를 출력
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


""" confusion matrix """

"""  """