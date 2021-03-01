import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

class LogisticRegressionClassifier(object):

    def __init__(self, max_iter=2000, learning_rate=0.001, lambda_=0.001):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.loss_ls = []
        self.lambda_ = lambda_

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def data_matrix(self, X):
        data_mat = []
        for d in X:
            data_mat.append([*d, 1])
        return data_mat

    def fit(self, X, y):
        data_mat = np.array(self.data_matrix(X))
        self.weights = np.zeros((data_mat.shape[-1], 1), dtype=np.float32)
        N = data_mat.shape[0]
        for iter_ in range(self.max_iter):
            y_hat = self.sigmoid(np.dot(data_mat, self.weights))
            loss = y.reshape(-1, 1) - y_hat
            self.loss_ls.append(-loss.mean())
            #             self.weights = self.weights +  self.learning_rate*np.dot(data_mat.T, loss)
            self.weights = self.weights * (1 - self.lambda_ * self.learning_rate) + self.learning_rate * np.dot(
                data_mat.T, loss)

    def plot_loss(self):
        plt.plot(list(range(self.max_iter)), self.loss_ls)
        plt.show()

    def score(self, X_test, y_test):
        right = 0
        X_test = self.data_matrix(X_test)
        for x, y in zip(X_test, y_test):
            result = np.dot(x, self.weights)[0]
            if (result > 0 and y == 1) or (result < 0 and y == 0):
                right += 1
        return right / len(X_test)
