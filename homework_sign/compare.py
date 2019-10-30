import os
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from homework_sign.optimizers import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 只显示error，不显示其他信息


def normalize(train):
    # 线性归一化
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - tf.reduce_min(x_data[i])) / (tf.reduce_max(x_data[i]) - tf.reduce_min(x_data[i]))
    return x_data.T


def norm_nonlinear(train):
    # 非线性归一化（log）
    x_data = train.T
    for i in range(4):
        x_data[i] = np.log10(x_data[i]) / np.log10(tf.reduce_max(x_data[i]))
    return x_data.T


def standardize(train):
    # 数据标准化（标准正态分布）
    x_data = train.T
    for i in range(4):
        x_data[i] = (x_data[i] - np.mean(x_data[i])) / np.std(x_data[i])
    return x_data.T


x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

x_data = standardize(x_data)

# 随机打乱数据
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

print("x.shape:", x_data.shape)
print("y.shape:", y_data.shape)
print("x.dtype:", x_data.dtype)
print("y.dtype:", y_data.dtype)
print("min of x:", tf.reduce_min(x_data))
print("max of x:", tf.reduce_max(x_data))
print("min of y:", tf.reduce_min(y_data))
print("max of y:", tf.reduce_max(y_data))

# from_tensor_slices函数切分传入的 Tensor 的第一个维度，生成相应的 dataset
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(10)

# seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样
w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))
variables = [w1, b1]
learning_rate_step = 10
learning_rate_decay = 0.8
train_loss_results = []
test_acc = []
lr = []
epoch = 500
loss_all = 0
learning_rate_base = 0.9
delta_w, delta_b = 0, 0
beta = 0.9
m_w, m_b = 0, 0
v_w, v_b = 0, 0
beta1, beta2 = 0.9, 0.999
global_step = tf.Variable(0, trainable=False)

for epoch in range(epoch):
    learning_rate = learning_rate_base * learning_rate_decay ** (epoch / learning_rate_step)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    lr.append(learning_rate)
    for step, (x_train, y_train) in enumerate(train_db):
        global_step = global_step.assign_add(1)

        with tf.GradientTape() as tape:
            y = tf.nn.softmax(tf.matmul(x_train, w1) + b1)
            y_onehot = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, y))
            # loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, y))
            loss_all += loss.numpy()
        # compute gradients
        grads = tape.gradient(loss, variables)
        # optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
        # w1, b1 = SGD(grads=grads, w=w1, b=b1, lr=learning_rate)
        # w1, b1 = SGD_M(grads=grads, w=w1, b=b1, lr=learning_rate, belta=beta, m_w=m_w, m_b=m_b)
        # w1, b1 = Adagrad(grads=grads, w=w1, b=b1, lr=learning_rate, v_w=v_w, v_b=v_b)
        # w1, b1, v_w, v_b = Adadelta(grads=grads, belta=beta, w=w1, b=b1, v_w=v_w, v_b=v_b, lr=learning_rate)
        w1, b1, m_w, m_b, v_w, v_b = Adam(global_step=global_step, grads=grads, belta1=beta1, belta2=beta2, w=w1, b=b1, v_w=v_w, v_b=v_b, m_w=m_w, m_b=m_b, lr=learning_rate)

        if step % 10 == 0:
            print("第{}次训练".format(epoch), 'loss:', float(loss))
        # print('m_w = ', m_w, '\nm_b = ', m_b, '\nv_w = ', v_w, '\nv_b = ', v_b)
            # print("lr=", learning_rate)
    train_loss_results.append(loss_all / 3)
    loss_all = 0

    # test(做测试）
    total_correct, total_number = 0, 0
    for step, (x_test, y_test) in enumerate(test_db):
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.sigmoid(y)

        pred = tf.argmax(y, axis=1)

        # 因为pred的dtype为int64，在计算correct时会出错，所以需要将它转化为int32
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int64)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
    acc = total_correct / total_number
    test_acc.append(acc)
    print("test_acc:", acc)
    print("---------------------")

# 绘制 loss 曲线
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label="$Loss$")
plt.legend()
plt.show()

# 绘制 Accuracy 曲线
plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()

# # 绘制 Learning_rate 曲线
# plt.title('Learning Rate Curve')
# plt.xlabel('Global steps')
# plt.ylabel('Learning rate')
# plt.plot(range(epoch + 1), lr, label="$lr$")
# plt.legend()
# plt.show()
