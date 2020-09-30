import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1) * 2. - 1. 
    x_test = x_test.reshape(-1, 28, 28, 1) * 2. - 1. 
   
    return x_train, x_test

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1) * 2. - 1. 
    x_test = x_test.reshape(-1, 28, 28, 1) * 2. - 1. 
   
    return x_train, x_test

def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255
    x_train = x_train * 2. - 1. 
    
    return x_train, x_test 
