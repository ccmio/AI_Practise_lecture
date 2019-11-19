import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras import Model, Sequential

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
checkpoint_save_path = './checkpoint/resnet18_cifar10_model.tf'
weight_path = './checkpoint/resnet18_weights.txt'
load_model = False

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)  # 给数据增加一个维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)

x_train, x_test = x_train / 255.0, x_test / 255.0
np.random.seed(1)
np.random.shuffle(x_train)
np.random.seed(1)
np.random.shuffle(y_train)

epochs = 5

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


class DecreaseDim(Model):
    def __init__(self, filters, shapes):
        super(DecreaseDim, self).__init__()
        self.filters = filters
        self.dd = Sequential([Conv2D(input_shape=shapes, filters=self.filters, kernel_size=(1, 1), strides=2, padding='valid', use_bias=False, kernel_initializer=tf.random_normal_initializer()),
                              BatchNormalization(),
                              Activation('relu')])

    def call(self, input):
        output = self.dd(input)
        return output


class ResNetBlock(Model):
    def __init__(self, filters, shapes, strides=1, residual_path=False):
        super(ResNetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path
        self.c1 = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=self.strides, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.c2 = Conv2D(filters=self.filters, kernel_size=(3, 3), strides=1, padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        if self.residual_path:
            self.dd = DecreaseDim(filters, shapes)

    def call(self, input):
        x = self.b1(input)
        x = self.a1(x)
        x = self.c1(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.c2(x)
        if self.residual_path:
            shortcut = self.dd(input)
            output = tf.add(x, shortcut)
        else:
            output = tf.add(x, input)
        return output


class ResNet(Model):
    def __init__(self, num_blocks):
        super(ResNet, self).__init__()

        self.conv_in = Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(7, 7), strides=2, padding='same')
        self.p_in = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')

        self.blocks = Sequential()
        for block_idx in range(num_blocks):
            filters = 64*np.power(2, block_idx)
            shapes = (8/np.power(2, block_idx), 8/np.power(2, block_idx), filters/2)
            for layer_idx in range(2):
                if block_idx > 0 and layer_idx == 0:
                    self.blocks.add(ResNetBlock(filters, shapes, strides=2, residual_path=True))
                else:
                    self.blocks.add(ResNetBlock(filters, shapes))

        self.p_out = GlobalAveragePooling2D()
        self.f = Flatten()
        self.dense = Dense(10, activation='softmax')

    def call(self, input):
        x = self.conv_in(input)
        x = self.p_in(x)
        x = self.blocks(x)
        x = self.p_out(x)
        x = self.f(x)
        output = self.dense(x)
        return output


model = ResNet(4)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

'''
*********************断点续训********************
'''
if load_model:
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 monitor='val_sparse_categorical_accuracy',
                                                 save_best_only=True,
                                                 verbose=1)

history = model.fit(image_gen_train.flow(x_train, y_train, batch_size=64),
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    validation_freq=1,
                    callbacks=[cp_callback],
                    verbose=1)
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

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(14, 8))

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

plt.savefig('/kaggle/working/loss_acc.jpg')
plt.show()