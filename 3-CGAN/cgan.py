import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class CGAN():
    def __init__( self, generator, discriminator
                , img_shape, noise_dim, label_dim):
    
        self.name = 'cGAN'
        self.gene = generator
        self.disc = discriminator
        self.img_shape = img_shape
        self.noise_dim = noise_dim
        self.label_dim = label_dim

    def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)):
       
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.gene.compile(loss='binary_crossentropy', optimizer=optimizer)
        
        self.disc.trainable = False
        input_noise = layers.Input(shape=self.noise_dim, name='input_noise')
        input_label = layers.Input(shape=self.label_dim, name='Input_label')
        fake_img = self.gene([input_noise, input_label])
        decision = self.disc([fake_img, input_label])
        
        self.cgan = models.Model( inputs=[input_noise, input_label]
                                , outputs=[decision])
        self.cgan.compile(loss="binary_crossentropy", optimizer=optimizer)

    def _make_datasets(self, x_data, y_data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(1)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
        return dataset

    def _make_constants(self, batch_size):
        zeros = tf.constant([[0.]] * batch_size)
        ones = tf.constant([[1.]] * batch_size)
        seed_noises = tf.random.normal(shape=[6*self.label_dim, self.noise_dim])
        seed_labels = [j for i in range(6) for j in range(self.label_dim)]
        seed_labels = to_categorical(seed_labels)
        return zeros, ones, seed_noises, seed_labels
    
    def _make_randoms(self, batch_size):
        random_noises = tf.random.normal(shape=[batch_size, self.noise_dim])
        random_labels = np.random.randint(0, self.label_dim, batch_size).reshape(-1,1)
        random_labels = to_categorical(random_labels, self.label_dim)
        return random_noises, random_labels 

    def train( self
             , x_data, y_data
             , epochs=1
             , batch_size=32
             , save_path=None):

        ## set data ##
        train_ds = self._make_datasets(x_data, y_data, batch_size)
        zeros, ones, seed_noises, seed_labels = self._make_constants(batch_size)

        ## epoch  ##
        history = {'d_loss':[], 'g_loss':[]}
        for epoch in range(1, 1+epochs):
            for h in history: history[h].append(0)
            
            ## batch-trainset ##
            for real_imgs, labels in train_ds:

                # phase 1 - train discriminator
                rnd_noises, _ = self._make_randoms(batch_size) 
                fake_imgs = self.gene.predict_on_batch([noises, labels])
                
                self.disc.trainable = True
                d_loss_real = self.disc.train_on_batch([real_imgs, labels], ones)
                d_loss_fake = self.disc.train_on_batch([fake_imgs, labels], zeros)
                d_loss = (0.5*d_loss_real) + (0.5*d_loss_fake)

                # phase 2 - train generator
                rnd_noises, rnd_labels = self._make_randoms(batch_size) 
                
                self.disc.trainable = False
                g_loss = self.cgan.train_on_batch([rnd_noises, rnd_labels], ones)
                
                history['d_loss'][-1]+=d_loss 
                history['g_loss'][-1]+=g_loss 
            
            print('* epoch: %i, d_loss: %f, g_loss: %f'%( epoch
                                                        , history['d_loss'][-1]
                                                        , history['g_loss'][-1]))

            self.plot_sample_imgs(seed_noises, seed_labels)

        return history
        

    def plot_sample_imgs(self, noises, labels, n=4, save_name=None):
        
        gen_imgs = self.gene.predict([noises, labels])
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)
            
        r, c = n, self.label_dim
        fig, axs = plt.subplots(r, c, figsize=(c,r))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray_r')
                axs[i,j].axis('off')
                cnt += 1

        if save_name!=None:
            fig.savefig(save_name)
        else: plt.show()
        plt.close()

    def plot_model(self, save_path):
        plot_model(self.cgan, to_file='%s/cgan.png'%save_path, show_shapes=True)
        plot_model(self.gene, to_file='%s/gene.png'%save_path, show_shapes=True)
        plot_model(self.disc, to_file='%s/disc.png'%save_path, show_shapes=True)

    def save_model(self, save_path):
        self.cgan.save('%s/cgan.h5'%save_path)
        self.gene.save('%s/gene.h5'%save_path)
        self.disc.save('%s/disc.h5'%save_path)

    def load_weights(self, weight_path):
       self.cgan.load_weights(weight_path)
