import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
checkpoint_save_path = './checkpoint/vgg16/saved_cifar10_model.tf'
weight_path = './checkpoint/vgg16/weights.txt'
load_model = False


'''
*********************加载mnist********************
'''
cifar10 = keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train, x_test = x_train / 255.0, x_test / 255.0
np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)



'''
*********************数据增强********************
'''
image_gen_train = ImageDataGenerator(
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放到50％
)
image_gen_train.fit(x_train)

'''
*********************网络定义********************
'''


class VGGNet(Model):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.c1 = Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')

        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)

        self.c3 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')

        self.c4 = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same')
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')

        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d2 = Dropout(0.2)

        self.c5 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')

        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b6 = BatchNormalization()
        self.a6 = Activation('relu')

        self.c7 = Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same')
        self.b7 = BatchNormalization()
        self.a7 = Activation('relu')

        self.p3 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d3 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b8 = BatchNormalization()
        self.a8 = Activation('relu')

        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b9 = BatchNormalization()
        self.a9 = Activation('relu')

        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b10 = BatchNormalization()
        self.a10 = Activation('relu')

        self.p4 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d4 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b11 = BatchNormalization()
        self.a11 = Activation('relu')

        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b12 = BatchNormalization()
        self.a12 = Activation('relu')

        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same')
        self.b13 = BatchNormalization()
        self.a13 = Activation('relu')

        self.p5 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
        self.d5 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(512, activation='sigmoid')
        self.d6 = Dropout(0.2)
        self.f2 = Dense(512, activation='sigmoid')
        self.d7 = Dropout(0.2)
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)
        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)
        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)
        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)
        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p5(x)
        x = self.d5(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d6(x)
        x = self.f2(x)
        x = self.d7(x)
        y = self.f3(x)
        return y


model = VGGNet()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

'''
*********************断点续训********************
'''
if load_model and os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, save_weights_only=True, monitor='loss',
                                              save_best_only=True, verbose=1)

h = model.fit(image_gen_train.flow(x_train, y_train, batch_size=32), epochs=1, validation_data=(x_test, y_test),
              validation_freq=1, callbacks=[cp_callback])
model.save_weights(checkpoint_save_path)

model.summary()

'''
*********************参数提取********************
'''
file = open(weight_path, 'w')  # 参数提取
for v in model.trainable_weights:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

'''
*********************loss/acc可视化********************
'''

acc = h.history['sparse_categorical_accuracy']
val_acc = h.history['val_sparse_categorical_accuracy']
loss = h.history['loss']
val_loss = h.history['val_loss']

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(acc, c='r', label='Training acc')
plt.plot(val_acc, c='b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()  # 绘制图例，默认在右上角

plt.subplot(1, 2, 2)
plt.plot(loss, c='r', label='Training loss')
plt.plot(val_loss, c='b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
