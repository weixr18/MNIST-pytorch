import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import platform
import struct
from time import perf_counter
'tensorflow version 2.x'


def show_pics(pics, num, rows=2):
    # 显示图片
    plt.figure()
    for i in range(num):
        plt.subplot(rows, (num // rows + 1), i + 1)
        plt.imshow(pics[i], cmap='bone')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 载入数据
def load_mnist():

    # 获取路径
    current_path = os.path.abspath(__file__)
    dir_path = os.path.abspath(
        os.path.dirname(current_path) + os.path.sep + ".")
    if platform.system() == 'Windows':
        dir_path += '\\..\\data'
    else:
        dir_path += '/../data'

    # 加载训练集
    labels_path = os.path.join(dir_path, 'train-labels-idx1-ubyte')
    images_path = os.path.join(dir_path, 'train-images-idx3-ubyte')

    # 训练标签
    with open(labels_path, 'rb') as lbpath:
        # >:big endian II:two unsigned ints
        _, n = struct.unpack('>II', lbpath.read(8))
        # labels
        train_labels = np.fromfile(lbpath, dtype=np.uint8)
        train_labels = tf.cast(train_labels, dtype=tf.int32)
        # 转为one-hot
        train_labels = tf.one_hot(train_labels, 10)

    # 训练数据
    with open(images_path, 'rb') as imgpath:
        _, __, ___, ____ = struct.unpack('>IIII', imgpath.read(16))
        train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(
            len(train_labels), 784)
        train_images = tf.cast(train_images, dtype=tf.float32) / 255.

    # 加载测试集
    labels_path = os.path.join(dir_path, 't10k-labels-idx1-ubyte')
    images_path = os.path.join(dir_path, 't10k-images-idx3-ubyte')

    # 训练标签
    with open(labels_path, 'rb') as lbpath:
        # >:big endian II:two unsigned ints
        _, n = struct.unpack('>II', lbpath.read(8))
        # labels
        test_labels = np.fromfile(lbpath, dtype=np.uint8)
        test_labels = tf.cast(test_labels, dtype=tf.int32)
        # 转为one-hot
        test_labels = tf.one_hot(test_labels, 10)

    # 训练数据
    with open(images_path, 'rb') as imgpath:
        _, __, ___, ____ = struct.unpack('>IIII', imgpath.read(16))
        test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(
            len(test_labels), 784)
        test_images = tf.cast(test_images, dtype=tf.float32) / 255.

    return (train_images, train_labels), (test_images, test_labels)


# 准备数据

(X, Y), (X_test, Y_test) = load_mnist()
batch_size = 100
train_db = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

# train_iter = iter(train_db)
# sample = next(train_iter)

# 准备参数
w1 = tf.Variable(tf.random.truncated_normal([28 * 28, 256],
                                            stddev=0.1))  # stddev 设置标准差 防止梯度弥散
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 10],
                                            stddev=0.1))  # stddev 设置标准差 防止梯度弥散
b2 = tf.Variable(tf.zeros([10]))

# 准备超参数
lr = 1e-3  # 固定学习率

start_t = perf_counter()
rand = 2957
rand_pics_f = X_test[rand:rand + 10]
rand_pics = rand_pics_f.numpy().reshape([10, 28, 28])
# show_pics(rand_pics, 10)

# 进行训练
for epoch in range(10):
    for step, (x, y) in enumerate(train_db):

        # 前向传播，计算损失
        with tf.GradientTape() as tape:
            h1 = tf.nn.leaky_relu(x @ w1 +
                                  tf.broadcast_to(b1, [x.shape[0], 256]))
            h2 = tf.math.softmax(h1 @ w2 +
                                 tf.broadcast_to(b2, [x.shape[0], 10]))
            log_h2 = tf.math.log(tf.clip_by_value(h2, 1e-8, 1.0))
            loss = -tf.reduce_sum(y * log_h2) / x.shape[0]

        # 计算梯度，更新参数
        grads = tape.gradient(loss, [w1, b1, w2, b2])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

        if step % 50 == 0:
            print('epoch:', epoch, 'step:', step, 'loss:', float(loss))

    # 进行验证
    def test():
        h1_t = tf.nn.leaky_relu(X_test @ w1 +
                                tf.broadcast_to(b1, [X_test.shape[0], 256]))
        h2_t = tf.math.softmax(h1_t @ w2 +
                               tf.broadcast_to(b2, [X_test.shape[0], 10]))

        h_tst_m = tf.argmax(h2_t, 1)
        Y_tst_m = tf.argmax(Y_test, 1)
        acc = tf.reduce_mean(tf.cast(tf.equal(h_tst_m, Y_tst_m), tf.float32))
        print("accuracy:", acc.numpy())

        h1_c = tf.nn.leaky_relu(rand_pics_f @ w1 +
                                tf.broadcast_to(b1, [10, 256]))
        h2_c = tf.math.softmax(h1_c @ w2 + tf.broadcast_to(b2, [10, 10]))
        pred = tf.argmax(h2_c, 1)
        print(pred.numpy())

    test()

print("total time:", perf_counter() - start_t)

# 检查
