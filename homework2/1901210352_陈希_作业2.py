import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 超参数设置
lr = 0.1  # 学习率
epoch = 10 # 训练次数

# 从iris.txt格式化读入features & labels
x_data = np.genfromtxt('./iris.txt', delimiter=',', dtype='float32')[:, :-1]
y_labels = np.genfromtxt('./iris.txt', delimiter=',', dtype=str)[:, -1]
y_data = np.array([0 if string == 'Iris-setosa' else 1 if string == 'Iris-versicolor' else 2 for string in y_labels])

# 打乱数据
np.random.seed(66666)
np.random.shuffle(x_data)
np.random.seed(66666)
np.random.shuffle(y_data)

# 确定训练测试集大小
x_train = x_data[:-30]
x_test = x_data[-30:]
y_train = y_data[:-30]
y_test = y_data[-30:]

# 形成训练集，测试集，设置每次训练batch大小
train_batch = 20
test_batch = 10
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(train_batch)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(test_batch)

# 初始化各层权重，设置隐层神经元数量，设置网络层数，学习率，训练次数等参数
stddev = 0.1
w1 = tf.Variable(tf.random.truncated_normal([4, 10], stddev=stddev, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([10], stddev=stddev, seed=1))
w2 = tf.Variable(tf.random.truncated_normal([10, 20], stddev=stddev, seed=2))
b2 = tf.Variable(tf.random.truncated_normal([20], stddev=stddev, seed=2))
w3 = tf.Variable(tf.random.truncated_normal([20, 10], stddev=stddev, seed=3))
b3 = tf.Variable(tf.random.truncated_normal([10], stddev=stddev, seed=3))
w4 = tf.Variable(tf.random.truncated_normal([10, 3], stddev=stddev, seed=4))
b4 = tf.Variable(tf.random.truncated_normal([3], stddev=stddev, seed=4))

# 循环训练epoch次
loss_all, loss = 0, 0
loss_line = []
for epoch in range(epoch):

    # 训练集输入模型
    for step, (train_db_x, train_db_y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(train_db_x, w1) + b1
            h2 = tf.matmul(h1, w2) + b2
            h3 = tf.matmul(h2, w3) + b3
            y = tf.matmul(h3, w4) + b4

            y_onehot = tf.one_hot(train_db_y, depth=3)
            loss = tf.reduce_mean(tf.square(y_onehot - y))
            loss_all += float(loss)

            # 梯度下降
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3, w4, b4])
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])
            w3.assign_sub(lr * grads[4])
            b3.assign_sub(lr * grads[5])
            w4.assign_sub(lr * grads[6])
            b4.assign_sub(lr * grads[7])
    loss_line.append(loss_all/(len(x_train)/train_batch))
    loss_all = 0

    # 测试集输入模型
    number_all, correct_all = 0, 0
    for (test_db_x, test_db_y) in test_db:
        h1 = tf.matmul(test_db_x, w1) + b1
        h2 = tf.matmul(h1, w2) + b2
        h3 = tf.matmul(h2, w3) + b3
        y = tf.matmul(h3, w4) + b4
        pred = tf.argmax(y, axis=1)

        correct = tf.reduce_sum(tf.cast(tf.equal(pred, test_db_y), dtype=tf.int64))
        correct_all += tf.cast(correct, dtype=tf.float64)
        number_all += tf.cast(len(test_db_x), dtype=tf.float64)
    acc = correct_all/number_all
    print('{:5d}次训练  loss:{:.8f} 准确率：{:.4f}'.format(epoch, float(loss), float(acc)))
print(w1, b1)
# 绘图
plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss_line)
plt.show()