from torch import no_grad



""" 평가 """
# 두 개의 분류에 대한 특징을 통해 테스트셋을 분류
@torch.no_grad()
def evaluation(test_loader):
    for x, y in test_loader
        X_test = x.test_data.view(-1, 28 * 28).float().to(device)
        Y_test = y.test_labels.to(device)

    """
        prediction = linear(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('single test Accuracy:', accuracy.item())

        # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
        r = random.randint(0, len(test) - 1)
        X_single_data = test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        Y_single_data = test.test_labels[r:r + 1].to(device)

    print('Label: ', Y_single_data.item())
    single_prediction = linear(X_single_data)
    print('Prediction: ', torch.argmax(single_prediction, 1).item())
    """


