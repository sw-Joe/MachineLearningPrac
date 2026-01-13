from ml_package.metric import Metrics

from torch import load, max, no_grad
import torch.cuda
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


"""  GPU 존재 확인 """
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    # torch.cuda.manual_seed_all(SEED)    # 난수 제어
    DEVICE = torch.device("cuda")


""" 모델 평가 """
def evaluation(model, model_status, test_loader, classes) -> None:
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
    print(f'Accuracy of the network on the test_image_set: {100 * correct // total} %')    # floor divi


def eval_confusion_matrix(model, model_status_PATH, test_loader, classes, time) -> None:
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


def eval_confusion_matrix_multiclass(model, model_status_PATH, test_loader, classes, time) -> None:
    """
    다중 클래스에 최적화된 혼동 행렬 시각화 및 평가 지표 출력
    """
    model.load_state_dict(torch.load(model_status_PATH, map_location=DEVICE))
    model.eval()

    y_true = []
    y_pred = []

    # 1. 예측 데이터 수집
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # 2. Confusion Matrix 계산
    cm = confusion_matrix(y_true, y_pred)
    # 각 행(Ground Truth)의 합으로 나누어 정규화된 행렬(비율) 생성
    cm_ratio = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # 3. 시각화 (크기를 10x8로 키워 가독성 확보)
    plt.figure(figsize=(10, 8))
    
    # annot 부분에 실제 개수(d)와 비율(.1%)을 함께 표시할 수 있도록 구성
    # (선택 사항: 복잡해 보인다면 fmt='d'로 유지해도 좋습니다)
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["({0:.1%})".format(value) for value in cm_ratio.flatten()]
    labels_combined = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels_combined = np.asarray(labels_combined).reshape(len(classes), len(classes))

    sns.heatmap(
        cm,
        annot=labels_combined, # 개수와 비율 병기
        fmt="",                # 문자열 형식을 그대로 사용
        cmap='Blues',          # 색상 가독성이 좋은 Blue 계열 유지
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Number of samples'}
    )
    
    plt.xlabel("Prediction", fontsize=12, fontweight='bold')
    plt.ylabel("Ground Truth", fontsize=12, fontweight='bold')
    plt.title(f"Confusion Matrix (Multi-class)\nTime: {time}", fontsize=15, pad=20)
    
    # 클래스 이름이 길 경우 겹치지 않게 회전
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{time}.png", dpi=150)
    plt.show()
    plt.close()

    # 4. Classification Report 출력 (상세 지표)
    print("\n[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=classes))


def visualize_classification_results(model, model_status_PATH, test_loader, classes, time, num_samples=4):
    """
    1. 전체 테스트 데이터셋에 대한 정확도를 정확히 산출
    2. Cat(좌측) -> Dog(우측) 순서로 성공/실패 사례 시각화
    3. 개별/평균 Confidence 수치 표시 및 텍스트 가독성 개선
    """
    # 모델 로드 및 평가 모드 설정
    model.load_state_dict(torch.load(model_status_PATH, map_location=DEVICE))
    model.eval()
    
    # 카테고리 맵핑 (실제, 예측) -> key
    cat_map = {
        (0, 0): 'cat_correct',
        (1, 0): 'cat_incorrect', # 실제 Dog인데 Cat으로 분류
        (1, 1): 'dog_correct',
        (0, 1): 'dog_incorrect'  # 실제 Cat인데 Dog으로 분류
    }
    
    # 데이터 저장을 위한 구조
    samples = {key: [] for key in cat_map.values()}
    confidences = {key: [] for key in cat_map.values()}
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    # 1. 전수 조사 루프 (정확도 계산 + 샘플 수집)
    with no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                lbl, pred = labels[i].item(), predicted[i].item()
                
                # 클래스별 정확도 누적 (전체 데이터 대상)
                class_total[classes[lbl]] += 1
                if lbl == pred:
                    class_correct[classes[lbl]] += 1
                
                # 시각화용 샘플 수집 (지정된 개수까지만)
                state_key = cat_map.get((lbl, pred))
                if state_key and len(samples[state_key]) < num_samples:
                    samples[state_key].append(imgs[i].cpu())
                    confidences[state_key].append(probs[i][pred].item() * 100)

    # 2. 정확도 최종 산출 (기존 evaluation 함수와 동일한 결과)
    accuracies = {
        cls: (100 * class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0)
        for cls in classes
    }

    # 3. 시각화 설정
    # 요청하신 순서: Cat 관련 열 좌측 배치
    categories = ['cat_correct', 'cat_incorrect', 'dog_correct', 'dog_incorrect']
    headers = [
        'Category', 
        'Cat TP\n(GT:Cat)', 'Cat FP\n(GT:Dog)', 
        'Dog TP\n(GT:Dog)', 'Dog FP\n(GT:Cat)'
    ]
    
    # (헤더 1행 + 샘플 n행 + 평균 1행)
    fig, axes = plt.subplots(num_samples + 2, 5, figsize=(18, 3.5 * (num_samples + 2)))
    
    # 헤더 행 작성
    for col, text in enumerate(headers):
        axes[0, col].text(0.5, 0.5, text, ha='center', va='center', fontsize=18, fontweight='bold')
        axes[0, col].axis('off')

    # 샘플 데이터 행 작성
    for row in range(num_samples):
        # 첫 번째 열: 샘플 번호
        axes[row + 1, 0].text(0.5, 0.5, f'Sample {row + 1}', ha='center', va='center', fontsize=16)
        axes[row + 1, 0].axis('off')
        
        for col, cat in enumerate(categories, start=1):
            ax = axes[row + 1, col]
            if row < len(samples[cat]):
                # 이미지 정규화 해제 및 클리핑
                img_disp = np.clip(samples[cat][row].permute(1, 2, 0).numpy(), 0, 1)
                ax.imshow(img_disp)
                ax.set_title(f"Conf: {confidences[cat][row]:.1f}%", fontsize=16, pad=8)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', color='gray', fontsize=14)
            ax.axis('off')

    # 마지막 행: 평균 확신도 (Avg Confidence)
    last_row = num_samples + 1
    axes[last_row, 0].text(0.5, 0.5, "Avg\nConfidence", ha='center', va='center', fontsize=17, fontweight='bold')
    axes[last_row, 0].axis('off')

    for col, cat in enumerate(categories, start=1):
        ax = axes[last_row, col]
        if confidences[cat]:
            avg_c = np.mean(confidences[cat])
            ax.text(0.5, 0.5, f"{avg_c:.1f}%", 
                    ha='center', va='center', color='blue', 
                    fontsize=20, fontweight='bold') # 텍스트 크기 강조
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', fontsize=16, color='gray')
        ax.axis('off')

    # 전체 제목 및 레이아웃 조정
    plt.suptitle(f'Classification Analysis: Cat Acc {accuracies["cat"]:.1f}% | Dog Acc {accuracies["dog"]:.1f}%', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 제목 공간 확보
    plt.savefig(f'classification_results_{time}.png', bbox_inches='tight', dpi=150)
    plt.show()

    print(f"✅ 시각화 완료: 'classification_results_{time}.png' 저장이 완료되었습니다.")
    return accuracies


def visualize_mnist_results(model, model_status_PATH, test_loader, time, num_samples=3):
    """
    MNIST(0-9) 분류 결과를 숫자별로 시각화합니다.
    각 숫자(Row)에 대해 성공 사례와 오답 사례를 보여줍니다.
    """
    model.load_state_dict(torch.load(model_status_PATH, map_location=DEVICE))
    model.eval()
    
    classes = [str(i) for i in range(10)]
    
    # 데이터 저장을 위한 구조: { '0': {'correct': [], 'incorrect': []}, ... }
    samples = {str(i): {'correct': [], 'incorrect': []} for i in range(10)}
    confidences = {str(i): {'correct': [], 'incorrect': []} for i in range(10)}
    
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    # 1. 전수 조사 및 샘플 수집
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                lbl_idx = labels[i].item()
                pred_idx = predicted[i].item()
                lbl_str = str(lbl_idx)
                
                # 정확도 누적
                class_total[lbl_str] += 1
                if lbl_idx == pred_idx:
                    class_correct[lbl_str] += 1
                    # 성공 샘플 수집
                    if len(samples[lbl_str]['correct']) < num_samples:
                        samples[lbl_str]['correct'].append(imgs[i].cpu())
                        confidences[lbl_str]['correct'].append(probs[i][pred_idx].item() * 100)
                else:
                    # 실패 샘플 수집 (실제 숫자가 lbl_idx인데 틀린 경우)
                    if len(samples[lbl_str]['incorrect']) < num_samples:
                        samples[lbl_str]['incorrect'].append((imgs[i].cpu(), pred_idx)) # 예측값도 저장
                        confidences[lbl_str]['incorrect'].append(probs[i][pred_idx].item() * 100)

    # 2. 시각화 설정 (10개 숫자 행 x [성공n개 + 실패n개] 열)
    # 열 구성: [숫자 라벨] + [Correct 샘플들] + [Incorrect 샘플들]
    total_cols = 1 + (num_samples * 2)
    fig, axes = plt.subplots(10, total_cols, figsize=(total_cols * 2.5, 20))
    
    for i in range(10):
        cls_str = str(i)
        acc = (100 * class_correct[cls_str] / class_total[cls_str]) if class_total[cls_str] > 0 else 0
        
        # 첫 번째 열: 숫자 정보 및 정확도
        axes[i, 0].text(0.5, 0.5, f"Digit: {i}\nAcc: {acc:.1f}%", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
        axes[i, 0].axis('off')

        # Correct 샘플 출력
        for s_idx in range(num_samples):
            col_idx = 1 + s_idx
            ax = axes[i, col_idx]
            if s_idx < len(samples[cls_str]['correct']):
                img = samples[cls_str]['correct'][s_idx].squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f"OK ({confidences[cls_str]['correct'][s_idx]:.1f}%)", fontsize=10, color='green')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
            ax.axis('off')

        # Incorrect 샘플 출력
        for s_idx in range(num_samples):
            col_idx = 1 + num_samples + s_idx
            ax = axes[i, col_idx]
            if s_idx < len(samples[cls_str]['incorrect']):
                img_data, pred_val = samples[cls_str]['incorrect'][s_idx]
                ax.imshow(img_data.squeeze().numpy(), cmap='magma') # 오답은 색상을 다르게 표시
                ax.set_title(f"Fail: {pred_val} ({confidences[cls_str]['incorrect'][s_idx]:.1f}%)", 
                             fontsize=10, color='red')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
            ax.axis('off')

    plt.suptitle(f'MNIST Classification Analysis (Total Accuracy inclusion)', fontsize=22, fontweight='bold', y=1.02)
    
    # 헤더 추가
    fig.text(0.3, 0.98, "Correct Samples (Success)", fontsize=16, color='green', fontweight='bold')
    fig.text(0.7, 0.98, "Incorrect Samples (Wrong Prediction)", fontsize=16, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'mnist_results_{time}.png', bbox_inches='tight', dpi=150)
    plt.show()

    return class_correct, class_total


import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_cifar10_results(model, model_status_PATH, test_loader, time, num_samples=3):
    """
    CIFAR-10 분류 결과를 클래스별로 시각화합니다.
    성공 사례는 정상 컬러로, 오답 사례는 시각적 구분을 위해 원본 컬러를 유지하되 타이틀로 강조합니다.
    """
    # 1. 모델 로드 및 설정
    model.load_state_dict(torch.load(model_status_PATH, map_location=DEVICE))
    model.eval()
    
    # CIFAR-10 클래스 정의
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 데이터 저장을 위한 구조
    samples = {cls: {'correct': [], 'incorrect': []} for cls in classes}
    confidences = {cls: {'correct': [], 'incorrect': []} for cls in classes}
    
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}

    # 2. 전수 조사 및 샘플 수집
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                lbl_idx = labels[i].item()
                pred_idx = predicted[i].item()
                lbl_str = classes[lbl_idx]
                pred_str = classes[pred_idx]
                
                # 정확도 누적
                class_total[lbl_str] += 1
                if lbl_idx == pred_idx:
                    class_correct[lbl_str] += 1
                    if len(samples[lbl_str]['correct']) < num_samples:
                        samples[lbl_str]['correct'].append(imgs[i].cpu())
                        confidences[lbl_str]['correct'].append(probs[i][pred_idx].item() * 100)
                else:
                    if len(samples[lbl_str]['incorrect']) < num_samples:
                        # 실패 샘플: 이미지와 함께 모델이 잘못 예측한 클래스 이름 저장
                        samples[lbl_str]['incorrect'].append((imgs[i].cpu(), pred_str)) 
                        confidences[lbl_str]['incorrect'].append(probs[i][pred_idx].item() * 100)

    # 3. 시각화 설정 (10개 클래스 행 x [성공n개 + 실패n개] 열)
    total_cols = 1 + (num_samples * 2)
    fig, axes = plt.subplots(10, total_cols, figsize=(total_cols * 3, 22))
    
    for i, cls_name in enumerate(classes):
        acc = (100 * class_correct[cls_name] / class_total[cls_name]) if class_total[cls_name] > 0 else 0
        
        # 첫 번째 열: 클래스 이름 및 정확도
        axes[i, 0].text(0.5, 0.5, f"{cls_name.upper()}\nAcc: {acc:.1f}%", 
                        ha='center', va='center', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Correct 샘플 출력
        for s_idx in range(num_samples):
            col_idx = 1 + s_idx
            ax = axes[i, col_idx]
            if s_idx < len(samples[cls_name]['correct']):
                # (C, H, W) -> (H, W, C)로 변환하여 matplotlib 출력
                img = samples[cls_name]['correct'][s_idx].permute(1, 2, 0).numpy()
                # 정규화 해제 (만약 Normalize를 적용했다면 원래대로 복구)
                img = np.clip(img, 0, 1) 
                ax.imshow(img)
                ax.set_title(f"OK ({confidences[cls_name]['correct'][s_idx]:.1f}%)", fontsize=9, color='green')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
            ax.axis('off')

        # Incorrect 샘플 출력
        for s_idx in range(num_samples):
            col_idx = 1 + num_samples + s_idx
            ax = axes[i, col_idx]
            if s_idx < len(samples[cls_name]['incorrect']):
                img_data, pred_name = samples[cls_name]['incorrect'][s_idx]
                img = img_data.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                # 오답은 빨간색 타이틀로 표시
                ax.set_title(f"Fail: {pred_name}\n({confidences[cls_name]['incorrect'][s_idx]:.1f}%)", 
                             fontsize=9, color='red')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='gray')
            ax.axis('off')

    plt.suptitle(f'CIFAR-10 Classification Analysis ({time})', fontsize=20, fontweight='bold', y=1.02)
    
    # 헤더 추가
    fig.text(0.35, 0.99, "Correct Samples (Success)", fontsize=15, color='green', fontweight='bold', ha='center')
    fig.text(0.75, 0.99, "Incorrect Samples (Wrong Prediction)", fontsize=15, color='red', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(f'cifar10_analysis_{time}.png', bbox_inches='tight', dpi=150)
    plt.show()

    return class_correct, class_total