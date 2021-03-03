# -*- coding: utf-8 -*-
import numpy as np
from keras.models import Model
from keras.datasets import mnist
from keras.models import load_model
from sklearn.neighbors import BallTree
from sklearn import preprocessing

topK = 10  # K: The number of nearest neighbors
avg_topK_acc = 0.

# Load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

# Load previsouly trained model
autoencoder = load_model('autoencoder.h5')

# Get encoder layer from trained model # 编码过程
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

print('Extracting features for training data ...')
train_codes = encoder.predict(x_train)  # embedding 特征
# 特征扁平
train_codes = train_codes.reshape(train_codes.shape[0],
                                  train_codes.shape[1] * train_codes.shape[2] * train_codes.shape[3])
# 0:样本数 1，2：特征图长宽 3：通道数
# L2-normalize the visual features to be Unit-norm.
train_codes = preprocessing.normalize(train_codes, norm='l2')

print('constructing index ...')  # 构造K近邻模型
tree = BallTree(train_codes, leaf_size=200)
# train_codes特征，leaf_size
# 改变leaf_size不会改变查询结果，但是会显著影响查询速度和存储内存。

print('Extracting features for testing data ...')
test_codes = encoder.predict(x_test)
test_codes = test_codes.reshape(test_codes.shape[0],
                                test_codes.shape[1] * test_codes.shape[2] * test_codes.shape[3])

# L2-normalize the visual features to be Unit-norm.
test_codes = preprocessing.normalize(test_codes, norm='l2')

# Begin to process each query
for i in range(test_codes.shape[0]):  # 遍历测试集
    if i % 1000 == 0:
        print('{} / {}'.format(i, test_codes.shape[0]))
    query_code = test_codes[i]
    query_label = y_test[i]  # 这个测试图片的label
    r_distances, r_index = tree.query([query_code], k=topK)
    # [query_code]： An array of points to query
    # K: The number of nearest neighbors to return
    # r_distances: each entry gives the list of distances to the neighbors of the corresponding point
    # r_index: each entry gives the list of indices of neighbors of the corresponding point
    r_labels = np.array([y_train[idx] for idx in r_index[0]])
    # nearest neighbors的label
    topK_acc = (r_labels == query_label).astype(int).sum() * 1. / topK
    avg_topK_acc += topK_acc
avg_topK_acc /= test_codes.shape[0]
print('The Averaged Top-{} Accuracy is: {}'.format(topK, avg_topK_acc))

# 参考资料
# http://lijiancheng0614.github.io/scikit-learn/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree.query
# https://www.jianshu.com/p/84d745b85fd5
# https://blog.csdn.net/pipisorry/article/details/53156836
