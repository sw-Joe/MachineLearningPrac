import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score)


class Metrics:
    def __init__(self, test, predict):
        self.test = test
        self.predict = predict


    def _accuracy(self):
        return accuracy_score(self.test, self.predict)
    

    def _precision(self):
        return precision_score(self.test, self.predict)
    

    def _recall(self):
        return recall_score(self.test, self.predict)

        
    def _f1(self):
        return f1_score(self.test, self.predict)


    def _roc_auc(self):
        return roc_auc_score(self.test, self.predict)
    

    def _confusion_matrix(self) -> None:
        sns.heatmap(confusion_matrix(self.test, self.predict), annot=True, fmt='d')
        plt.xlabel('Prediction')
        plt.ylabel('Real')
        plt.savefig('confusion_matrix.png')
        return
    
    # MAE, MSE, RMSE, MAPE