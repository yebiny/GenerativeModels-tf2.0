import numpy as np
from buildModel import *
from drawTools import *

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
    y0 = tf.constant([[0.]] * batch_size)
    y1 = tf.constant([[1.]] * batch_size)
    seed = tf.random.normal(shape=[16, noise_dim])
    return y0, y1, seed


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

    
    def train(self, n_epochs):

        # Set data
        dataset = get_dataset(self.x_data, self.batch_size)

        # y1 : half '0' half '1' for discriminator train 
        # y2 : all '1' for generator train
        y0, y1, seed = make_constants(self.noise_dim, self.batch_size)
        plot_generated_images(self.generator, seed, save=self.save_path+'/generatedImg_0')

        # Train
        history = {'epoch':[], 'd_loss':[], 'g_loss':[]}
        for epoch in range(1, n_epochs+1):
            print("Epoch {}/{}".format(epoch, n_epochs))
            
            d_loss=g_loss=0
            for images in dataset:
                # phase 1 - training the discriminator
                noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
                generated_images = self.generator.predict(noise)

                self.discriminator.trainable = True
                d_loss = d_loss+0.5*self.discriminator.train_on_batch(images, y1)
                d_loss = d_loss+0.5*self.discriminator.train_on_batch(generated_images, y0)

                # phase 2 - training the generator
                noise = tf.random.normal(shape=[self.batch_size, self.noise_dim])
                self.discriminator.trainable = False
                g_loss = g_loss+self.gan.train_on_batch(noise, y1)
            
            print('d_loss:%f, g_loss:%f'%(d_loss, g_loss))
            history['epoch'].append(epoch)
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)
            plot_generated_images(self.generator, seed, save=self.save_path+'/generatedImg_%i'%epoch)
            
        return history
    
