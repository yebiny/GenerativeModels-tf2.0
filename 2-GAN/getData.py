import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10

def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train * 2. - 1. 
    print(x_train.shape) 
    
    return x_train

def get_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_train = x_train * 2. - 1. 
    print(x_train.shape) 
   
    return x_train

def get_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train * 2. - 1. 
    print(x_train.shape) 

    return x_train 
