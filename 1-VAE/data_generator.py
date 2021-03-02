import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical

def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    return x_train

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    return x_train

def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255
    return x_train

def get_celeba(img_size, n_dataset=10000):
    dataset, metadata = tfds.load('celeb_a', with_info=True)
    train=dataset['train']
    x_data = []
    for i, t in enumerate(train):
        img = t['image']
        img = tf.image.resize(img, [img_size+20, img_size+20], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = (np.array(img)/255)

        x_data.append(img[10:img_size+10,10:img_size+10,:])

        if i>=n_dataset-1: break

    return np.array(x_data)
