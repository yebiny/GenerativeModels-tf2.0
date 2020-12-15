import numpy as np
from buildModel import *
from drawTools import *
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import *

class GAN():
    def __init__(self, x_data, save_path, z_dim=100, batch_size=32):
    
        self.x_data = x_data
        self.save_path=save_path
        self.z_dim = z_dim
        self.batch_size= batch_size

        self.gan, self.generator, self.discriminator= build_gan(x_data.shape, z_dim)
        plot_model(self.gan, to_file=self.save_path+'/gan.png', show_shapes=True)
        plot_model(self.generator, to_file=self.save_path+'/generator.png',show_shapes=True)
        plot_model(self.discriminator, to_file=self.save_path+'/discriminator.png',show_shapes=True)

    def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)):
        # When compile generator(gan), discriminator must not trainable!
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.discriminator.trainable = False
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)


    def make_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.x_data).shuffle(1)
        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(1)
        return dataset

    def make_constants(self):
        y0 = tf.constant([[0.]] * self.batch_size)
        y1 = tf.constant([[1.]] * self.batch_size)
        seed = tf.random.normal(shape=[self.batch_size, self.z_dim])
        return y0, y1, seed
   
    def make_randoms(self):
        random_noises =  tf.random.normal(shape=[self.batch_size, self.z_dim])
        return random_noises

    def train(self, n_epochs):

        # Set data
        dataset = self.make_datasets()

        # y1 : all value '0' and shape is (batch_size, )
        # y2 : all value '1' and shape is (bathc_size, )
        # seed : random values and shape is ( batch_size * z_dimension )
        y0, y1, seed = self.make_constants()
        plot_generated_images(self.generator, seed, save=self.save_path+'/generatedImg_0')

        # Train
        history = {'epoch':[], 'd_loss':[], 'g_loss':[]}
        for epoch in range(1, n_epochs+1):
            print("Epoch {}/{}".format(epoch, n_epochs))
            
            d_loss=g_loss=0
            for x_real in dataset:
                # phase 1 - training the discriminator
                random_noises = self.make_randoms()
                x_fake = self.generator.predict(random_noises)
                
                self.discriminator.trainable = True
                dl1 = self.discriminator.train_on_batch(x_real, y1)
                dl2 = self.discriminator.train_on_batch(x_fake, y0)
                d_loss = d_loss + (0.5*dl1) + (0.5*dl2)

                # phase 2 - training the generator
                random_noises = self.make_randoms()

                self.discriminator.trainable = False
                gl = self.gan.train_on_batch(random_noises , y1)
                g_loss = g_loss + gl
            
            print('d_loss:%f, g_loss:%f'%(d_loss, g_loss))
            history['epoch'].append(epoch)
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)
            plot_generated_images(self.generator, seed, save=self.save_path+'/generatedImg_%i'%epoch)
            
        return history
    
