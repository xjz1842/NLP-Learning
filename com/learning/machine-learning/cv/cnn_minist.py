# Keras官方给出的结果
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from tensorboard.program import TensorBoard

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train[0].shape)


def load_data(path):
    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

print("x_test:",x_test[2000:])

print(input_shape)
x_train, x_test = x_train.astype('float32'), x_test.astype('float32')
print(x_train[0, :1, :2])

x_train, x_test = x_train / 255, x_test / 255
print("x_train.shape:", x_train.shape)
print("train samples:", x_train.shape[0])
print("test sampels:", x_test.shape[0])

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape)
print(y_train)
print("true label is: ", np.where(y_test[:1][0] == 1)[0])

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.1),
              #               optimizer=keras.optimizers.Adadelta(),
              metrics=["accuracy"]
              )

print(model.summary())

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test[2000:], y_test[2000:])
                    )

score = model.evaluate(x_test[2000:], y_test[2000:], verbose=1)
print("test accuracy:", score[1])

model.save("./mnist.h5")

model_mnist = keras.models.load_model("mnist.h5")

result = model_mnist.predict_classes(x_test[:1], batch_size=1, verbose=0)
print("predict is: ", result)
print("true label is: ", np.where(y_test[:1][0] == 1)[0])
