import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical

def get_fashion_mnist():
    (x_data, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data.reshape(-1, 28, 28, 1)
    return x_data

def get_mnist():
    (x_data, y_train), (x_test, y_test) = mnist.load_data()
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data.reshape(-1, 28, 28, 1)
    return x_data

def get_cifar10():
    (x_data, y_train), (x_test, y_test) = cifar10.load_data()
    x_data = x_data.astype(np.float32) / 255
    return x_data

def get_celeba(img_size, n_dataset=10000):
    dataset, metadata = tfds.load('celeb_a', with_info=True)
    train_ds=dataset['train']

    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    x_data = []
    for i, train in enumerate(train_ds):
        img = train['image']
        img = tf.image.resize(img, [img_size+20, img_size+20], method=method)
        img = (np.array(img)/255)

        x_data.append(img[10:img_size+10,10:img_size+10,:])

        if i>=n_dataset-1: break

    return np.array(x_data)
