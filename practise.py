import os
from sklearn import datasets
import numpy as np
from keras.datasets import fashion_mnist
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Flatten
from keras import Model
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

""" 鸢尾花训练

x = datasets.load_iris().data
y = datasets.load_iris().target

x_train = x[:120]
y_train = y[:120]
x_test = x[120:]
y_test = y[120:]
np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)


class Iris_Model(Model):
    def __init__(self):
        super(Iris_Model, self).__init__()
        self.dense = Dense(3, activation='softmax')


    def call(self, x):
        y = self.dense(x)
        return y


model = Iris_Model()
model.compile(optimizer=tf.optimizers.SGD(lr=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, epochs=500, validation_freq=20, validation_data=(x_test, y_test))
model.summary()

"""

mnist = fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)
np.random.seed(2)
np.random.shuffle(x_test)
np.random.seed(2)
np.random.shuffle(y_test)


class MnistModel(Model):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flattern = Flatten(input_shape=(28, 28))
        self.dense1 = Dense(256, activation='relu')
        self.dense2 = Dense(10, activation='softmax')

    def call(self, x):
        y = self.flattern(x)
        y = self.dense1(y)
        y = self.dense2(y)
        return y


model = MnistModel()
model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_freq=1, validation_data=(x_test, y_test))

plt.figure()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = history.epoch

plt.plot(epochs, acc, c='r', label='Training acc')
plt.plot(epochs, val_acc, c='b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()  # 绘制图例，默认在右上角

plt.figure()

plt.plot(epochs, loss, c='r', label='Training loss')
plt.plot(epochs, val_loss, c='b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
