import os, sys
import argparse
import numpy as np
from tensorflow.keras.utils import plot_model
from buildModel import *
from drawTools import *
from getData import *


def plot_generated_images(generator, seed, save=None):
    generated_images = generator.predict(seed)
    generated_images = (generated_images+1)/2
    plot_multiple_images(generated_images, 4, save) 

def get_dataset(x_data, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
    print(dataset)
    return dataset

def make_constants(noise_dim, batch_size):
    y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
    y2 = tf.constant([[1.]] * batch_size)
    seed = tf.random.normal(shape=[16, noise_dim])
    return y1, y2, seed


class GAN():
    def __init__(self, x_data, save_path, noise_dim=100, batch_size=32):
    
        self.x_data = x_data
        self.save_path=save_path
        self.noise_dim = noise_dim
        self.batch_size= batch_size

        self.gan = build_gan(x_data.shape, noise_dim)
        self.generator, self.discriminator = self.gan.layers
        self.gan.summary()

        # When compile generator(gan), discriminator must not trainable!
        self.discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
        self.discriminator.trainable = False
        self.gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

    
    def fit(self, n_epochs):

        # Set data
        dataset = get_dataset(self.x_data, self.batch_size)

        # y1 : half '0' half '1' for discriminator train 
        # y2 : all '1' for generator train
        y1, y2, seed = make_constants(self.noise_dim, self.batch_size)
        plot_generated_images(self.generator, seed, save=self.save_path+'/generatedImg')

        # Train
        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch + 1, n_epochs))
            for images in dataset:
                # phase 1 - training the discriminator
                noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
                generated_images = self.generator.predict(noise)

                self.discriminator.trainable = True
                self.discriminator.train_on_batch(tf.concat([generated_images, images], axis=0), y1)

                # phase 2 - training the generator
                noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
                self.discriminator.trainable = False
                self.gan.train_on_batch(noise, y2)

            plot_generated_images(self.generator, seed, save=self.save_path+'/generatedImg_%i'%epoch)


    
def main():
    opt = argparse.ArgumentParser(description="==== GAN with tensorflow2.x ====")
    opt.add_argument(dest='save_path', type=str, help=': set save directory ')
    opt.add_argument('--data', type=str, default='fmnist', help=': choice among [mnist / fmnist / cifar] (default: [fmnist] )')
    opt.add_argument('-e',  dest='epochs', type=int, default=5, help=': number epochs')
    
    argv = opt.parse_args()
    if not os.path.exists(argv.save_path): os.makedirs(argv.save_path)
    
    if argv.data =='mnist':
        x_train, _ = get_mnist()
    elif argv.data =='fmnist':
        x_train, _ = get_fashion_mnist()
    elif argv.data =='cifar':
        x_train, _ = get_cifar10()
    print('* Use dataset', argv.data, x_train.shape)
    
    gan = GAN(x_train, argv.save_path)
    plot_model(gan.gan, to_file=argv.save_path+'/gan.png', show_shapes=True)
    plot_model(gan.generator, to_file=argv.save_path+'/gene.png',show_shapes=True)
    plot_model(gan.discriminator, to_file=argv.save_path+'/disc.png',show_shapes=True)

    gan.fit(argv.epochs)

if __name__=='__main__':
    main()

