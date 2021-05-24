import numpy as np
import tensorflow as tf
try:
    from tensorflow.python import keras
except:
    from tensorflow import keras

import os
import platform
import struct
import matplotlib.pyplot as plt
'tensorflow version 2.x'


def show_pics(pics, num, rows=2):
    # 显示图片
    plt.figure()
    for i in range(num):
        plt.subplot(rows, (num // rows), i + 1)
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
        _, __ = struct.unpack('>II', lbpath.read(8))
        # labels
        train_labels = np.fromfile(lbpath, dtype=np.uint8)
        train_labels = tf.cast(train_labels, dtype=tf.int32)
        # 转为one-hot
        # train_labels = tf.one_hot(train_labels, 10)

    # 训练数据
    with open(images_path, 'rb') as imgpath:
        _, __, ___, ____ = struct.unpack('>IIII', imgpath.read(16))
        train_images = np.fromfile(imgpath, dtype=np.uint8).reshape(
            len(train_labels), 784)
        train_images = tf.cast(train_images, dtype=tf.float32) / 255.

    # 加载测试集
    labels_path = os.path.join(dir_path, 't10k-labels-idx1-ubyte')
    images_path = os.path.join(dir_path, 't10k-images-idx3-ubyte')

    # 测试标签
    with open(labels_path, 'rb') as lbpath:
        # >:big endian II:two unsigned ints
        _, __ = struct.unpack('>II', lbpath.read(8))
        # labels
        test_labels = np.fromfile(lbpath, dtype=np.uint8)
        test_labels = tf.cast(test_labels, dtype=tf.int32)
        # 转为one-hot
        # test_labels = tf.one_hot(test_labels, 10)

    # 测试数据
    with open(images_path, 'rb') as imgpath:
        _, __, ___, ____ = struct.unpack('>IIII', imgpath.read(16))
        test_images = np.fromfile(imgpath, dtype=np.uint8).reshape(
            len(test_labels), 784)
        test_images = tf.cast(test_images, dtype=tf.float32) / 255.

    return (train_images, train_labels), (test_images, test_labels)


# 准备数据
batch_size = 100
(X, Y), (X_test, Y_test) = load_mnist()
X = tf.reshape(X, [60000, 28, 28, 1])
X_test = tf.reshape(X_test, [10000, 28, 28, 1])
train_db = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)

# 准备超参数
conv1_filters = 32  # 第一层卷积层卷积核的数目
conv2_filters = 64  # 第二层卷积层卷积核的数目
fc1_units = 1024  # 第一层全连接层神经元的数目
LR = 0.001

# 定义网络
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(conv1_filters, (5, 5),
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(conv2_filters, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 指定优化器和损失函数
model.compile(optimizer=tf.optimizers.Adam(LR),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""

# 训练
history = model.fit(X, Y, epochs=5)

# 测试
loss, acc = model.evaluate(X_test, Y_test)
print("train model, accuracy:{:5.2f}%".format(100 * acc))

# 保存权重
model.save_weights('./save_weights/save_test')
"""

# 恢复权重
model.load_weights('./save_weights/save_test')

# 测试模型
# loss, acc = model.evaluate(X_test, Y_test)
# print("Restored model, accuracy:{:5.2f}%".format(100 * acc))

rand = 2957
TEST_NUM = 30

rand_pics_f = X_test[rand:rand + TEST_NUM]
rand_pics = rand_pics_f.numpy().reshape([TEST_NUM, 28, 28, 1])

prediction = model.predict(x=rand_pics)
print("\n\n\n",
      tf.argmax(prediction, 1).numpy().reshape([5, TEST_NUM // 5]), "\n\n\n")

rand_pics = rand_pics.reshape([TEST_NUM, 28, 28])
show_pics(rand_pics, TEST_NUM, 5)

# """