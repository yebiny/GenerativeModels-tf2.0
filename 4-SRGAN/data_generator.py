import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
def get_celeba(n_dataset, hr_size, lr_size):
    dataset, metadata = tfds.load('celeb_a', with_info=True)
    train=dataset['train']
    hr_out = []
    lr_out = []
    for i, t in enumerate(train):
        img = t['image']
        hr_img = tf.image.resize(img, [hr_size, hr_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        lr_img = tf.image.resize(img, [lr_size, lr_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        hr_img = (np.array(hr_img)/127.5)-1
        lr_img = (np.array(lr_img)/127.5)-1

        hr_out.append(hr_img)
        lr_out.append(lr_img)

        if i>=n_dataset-1: break

    return np.array(hr_out), np.array(lr_out)
