import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class GAN():
    def __init__(self, generator, discriminator, img_shape, noise_dim):
        
        self.name = 'GAN'
        self.gene = generator
        self.disc = discriminator
        self.img_shape = img_shape
        self.noise_dim = noise_dim
    
    def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)):
        
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.gene.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.disc.trainable = False
        input_noise = layers.Input(shape=self.noise_dim, name='input_noise')
        fake_img = self.gene(input_noise)
        decision = self.disc(fake_img)
        
        self.gan = models.Model(inputs=input_noise, outputs=decision)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    def _make_datasets(self, x_data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(x_data).shuffle(1)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
        return dataset

    def _make_constants(self, size):
        zeros = tf.constant([[0.]] * size)
        ones = tf.constant([[1.]] * size)
        seed = tf.random.normal(shape=[size, self.noise_dim])
        return zeros, ones, seed
   
    def _make_random_noises(self, size):
        random_noises =  tf.random.normal(shape=[size, self.noise_dim])
        return random_noises
    

    def train( self
             , x_data
             , epochs=1
             , batch_size=32
             , save_path=None):
        
        # set data
        train_ds = self._make_datasets(x_data, batch_size)
        zeros, ones, seed_noises = self._make_constants(batch_size)
        
        # epoch
        history = {'d_loss':[], 'g_loss':[]}
        for epoch in range(1, 1+epochs):
            for h in history: history[h].append(0)
        
            # batch-trainset
            for real_imgs in train_ds:
                
                # phase 1 - training the discriminator
                rnd_noises = self._make_random_noises(batch_size)
                fake_imgs = self.gene.predict_on_batch(rnd_noises)
                
                self.disc.trainable = True
                d_loss_real = self.disc.train_on_batch(real_imgs, ones)
                d_loss_fake = self.disc.train_on_batch(fake_imgs, zeros)
                d_loss = (0.5*d_loss_real) + (0.5*d_loss_fake)
                
                # phase 2 - training the generator
                rnd_noises = self._make_random_noises(batch_size)
                
                self.disc.trainable = False
                g_loss = self.gan.train_on_batch(rnd_noises , ones)
                
                history['d_loss'][-1]+=d_loss
                history['g_loss'][-1]+=g_loss
            
            print('* epoch: %i, d_loss: %f, g_loss: %f'%( epoch
                                                        , history['d_loss'][-1]
                                                        , history['g_loss'][-1]))
            
            self.plot_sample_imgs(seed_noises) 
        
        return history

    def plot_sample_imgs(self, noises, n=8, save_name=None):
        plt.figure(figsize=(n,2))
        gen_imgs = self.gene.predict(noises[:2*n])
        gen_imgs = 0.5 * (gen_imgs+1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        for i, img in enumerate(gen_imgs):
            plt.subplot(2,n,i+1)
            plt.imshow(np.squeeze(img), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])

        if save_name!=None:
            plt.savefig(save_path)
        else: plt.show()

    def plot_model(self, save_path):
        plot_model(self.gan, to_file=self.save_path+'/gan.png', show_shapes=True)
        plot_model(self.gene, to_file=self.save_path+'/gene.png',show_shapes=True)
        plot_model(self.disc, to_file=self.save_path+'/disc.png',show_shapes=True)

    def save_model(self, save_path):
        self.gan.save('%s/gan.h5'%save_path)
        self.gene.save('%s/gene.h5'%save_path)
        self.disc.save('%s/disc.h5'%save_path)

    def load_weights(self, weight_path):
        self.cgan.load_weights(weight_path)
