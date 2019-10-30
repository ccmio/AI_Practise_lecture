import os
import numpy as np
from keras.datasets import fashion_mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.models import Model, Sequential
import matplotlib.pyplot as plt
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息

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


x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
x_train = np.array(x_train, dtype='float32')
x_test = np.array(x_test, dtype='float32')

model = Sequential()
'''
92.21%
'''
model.add(Conv2D(input_shape=(28, 28, 1),
                            kernel_size=(3, 3),
                            filters=30,
                            padding='same',
                            activation='relu',))
model.add(MaxPool2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='Adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_freq=1, validation_data=(x_test, y_test))
model.summary()

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

model.save('mymodel.h5')
model.save_weights('mymodel_weight.h5')