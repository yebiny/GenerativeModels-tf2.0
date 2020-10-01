import numpy as np
from buildModel import *
from drawTools import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam

class GAN():
    def __init__(self, x_data, save_path, noise_dim=100, batch_size=32):
    
        self.x_data = x_data
        self.save_path=save_path
        self.noise_dim = noise_dim
        self.batch_size= batch_size

        self.gan, self.generator, self.discriminator= build_gan(x_data.shape, noise_dim)
        plot_model(self.gan, to_file=self.save_path+'/gan.png', show_shapes=True)
        plot_model(self.generator, to_file=self.save_path+'/generator.png',show_shapes=True)
        plot_model(self.discriminator, to_file=self.save_path+'/discriminator.png',show_shapes=True)

    def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)):
        # When compile generator(gan), discriminator must not trainable!
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.discriminator.trainable = False
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def make_constants(self):
        y0 = tf.constant([[0.]] * self.batch_size)
        y1 = tf.constant([[1.]] * self.batch_size)
        seed = tf.random.normal(shape=[self.batch_size, self.noise_dim])
        return y0, y1, seed
    
    def train(self, n_epochs):

        # Set data
        dataset = tf.data.Dataset.from_tensor_slices(self.x_data).shuffle(1000)
        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(1)

        # y1 : half '0' half '1' for discriminator train 
        # y2 : all '1' for generator train
        y0, y1, seed = self.make_constants()
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
    
