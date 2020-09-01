import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os, sys

def if_not_exit(path):
    if not os.path.exists(path):
        print(path, 'is not exist.')
        exit()
def if_exist(path):
    if os.path.exists(path):
        print(path, 'is exist. Save at other directory.')
        exit()

def if_not_make(path):
    if not os.path.exists(path):
        os.makedirs(path)

def data_generator(x_data, y_data, batch_size):
    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data,  train_size = 0.8, random_state=34)
    x_valid, x_test, y_valid, y_test = train_test_split(x_valid, y_valid,  train_size = 0.8, random_state=34)
    
    print(x_train.shape, y_train.shape) 
    print(x_valid.shape, y_valid.shape) 
    print(x_test.shape, y_test.shape)

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def reduce_lr(pre_v_loss, v_loss, count, lr, patience, factor, min_lr):
    if v_loss < pre_v_loss:
        count = 0
    else:
        count += 1
        if count >= patience: 
            lr = lr*factor
            if lr < min_lr: 
                lr = min_lr
            count = 0
            print('reduce learning rate..', lr)    
    return count, lr

def draw_recimg(model, data, epoch, duration, save_path):
    if epoch%duration == 0:
        org = data[:100]
        rec = model(data[:100], training=False)
        fig, ax = plt.subplots(6, 10, figsize=(20, 10))
        for i in range(10):
            for j in range(6):
                if j%2 ==0: img=org
                else: img=rec
                ax[j][i].set_axis_off()
                ax[j][i].imshow(img[10*(j//2)+i])

        plt.savefig('%s/recimg_%i.png'%(save_path, epoch))
        plt.close('all')
