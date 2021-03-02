import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.utils import to_categorical

def get_fashion_mnist():
    (x_data, y_data), (x_test, y_test) = fashion_mnist.load_data()
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data.reshape(-1, 28, 28, 1)
    x_data = x_data* 2. - 1. 
    
    y_data = to_categorical(y_data, num_classes=max(y_data)+1)
    return x_data, y_data

def get_mnist():
    (x_data, y_data), (x_test, y_test) = mnist.load_data()
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data.reshape(-1, 28, 28, 1)
    x_data = x_data* 2. - 1. 
    
    y_data = to_categorical(y_data, num_classes=max(y_data)+1)
    return x_data, y_data

def get_cifar10():
    (x_data, y_data), (x_test, y_test) = cifar10.load_data()
    x_data = x_data.astype(np.float32) / 255
    x_data = x_data * 2. - 1. 
    
    y_data = np.reshape(y_data, len(y_data))
    y_data = to_categorical(y_data, num_classes=max(y_data)+1)
    return x_data, y_data 
