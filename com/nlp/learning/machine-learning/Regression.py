from sklearn import linear_model

import random


def get_data(w, num):
    x = [random.uniform(0, 5) for i in range(0, num)]
    y = [w * s for s in x]
    return zip(x, y)


def train_step_pow(data, w, rate=0.03):
    g = sum([(w * x - y) * x for [x, y] in data]) / len(data)
    w = w - rate * g
    return w


def train_step_abs(data, w, rate=0.03):
    g = sum(x if (w * x - y) > 0 else - 1 * x for [x, y] in data) / len(data)
    w = w - rate * g
    return w


def call_data_error(data, w):
    error = [(w * x - y) * (w * x - y) for [x, y] in data]
    return error


def linear():
    clf = linear_model.LinearRegression()
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]

    clf.fit(X, y)

    print(clf.coef_)
    print(clf.intercept_)


if __name__ == "__main__":
    data = list(get_data(10, 10)) + list(get_data(6, 2))
    print(list(data))

    w1 = w2 = 7

    for i in range(0, 5000):
        w1 = train_step_pow(data, w1)
        w2 = train_step_abs(data, w2)

        if i % 50 == 0:
            print('{} {}'.format(w1, w2))
