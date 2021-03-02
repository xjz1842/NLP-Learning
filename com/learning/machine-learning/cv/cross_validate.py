import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
from LogisticRegressionClassifier import LogisticRegressionClassifier

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    #   取前100行：两种花 [0,1,-1]：第一列 第二列两个特征和最后一列label
    data = np.array(df.iloc[:100, [0, 1, -1]])
    #     返回特征和标签
    return data[:, :2], data[:, -1]

X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=True)
# c_score = []
# Cs = np.logspace(-2, 4, num=100)
# shape = X_train.shape[0]
# Kfold = 5
# score = []
# for c in Cs:
#     for k in range(Kfold):
#         X_train_ = np.concatenate((X_train[0:k, ], X_train[k + int(shape / Kfold):, ]), axis=0)
#         y_train_ = np.concatenate((y_train[0:k], y_train[k + int(shape / Kfold):]), axis=0)
#         X_test_, y_test_ = X_train[k:k + int(shape / Kfold), ], y_train[k:k + int(shape / Kfold)]
#         lr_clf = LogisticRegressionClassifier(lambda_=float(1 / c))
#         lr_clf.fit(X_train_, y_train_)
#         score.append(lr_clf.score(X_test_, y_test_))
#     c_score.append(np.mean(score))
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(Cs, c_score)
# ax.set_xscale('log')
# ax.set_xlabel("C")
# ax.set_ylabel("acc")
# plt.show()

from sklearn.model_selection import cross_val_score
from sklearn import linear_model
c_score = []
Cs=np.logspace(-2,4,num=100)
for c in Cs:
    lr = linear_model.LogisticRegression(C=c)
    score = cross_val_score(lr,X_train,y_train,cv=5,scoring='accuracy')
    c_score.append(score.mean())
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(Cs,c_score)
ax.set_xscale('log')
ax.set_xlabel("C")
ax.set_ylabel("acc")
plt.show()