import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class CGAN():
    def __init__( self, generator, discriminator
                , img_shape, label_dim, noise_dim):
    
        self.name = 'cGAN'
        self.gene = generator
        self.disc = discriminator
        self.img_shape = img_shape
        self.label_dim = label_dim
        self.noise_dim = noise_dim

    def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)):
       
        self.disc.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.disc.trainable = False
        
        input_noise = layers.Input(shape=self.noise_dim, name='input_noise')
        input_label = layers.Input(shape=self.label_dim, name='Input_label')
        
        fake_img = self.gene([input_noise, input_label])
        decision = self.disc([fake_img, input_label])
        self.cgan = models.Model( inputs=[input_noise, input_label]
                                , outputs=[decision])
        self.cgan.compile(loss="binary_crossentropy", optimizer=optimizer)
        self.disc.trainable=True

    def _make_datasets(self, x_data, y_data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data)).shuffle(1)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
        return dataset

    def _make_constants(self, batch_size):
        y0 = tf.constant([[0.]] * batch_size)
        y1 = tf.constant([[1.]] * batch_size)
        seed_noises = tf.random.normal(shape=[6*self.label_dim, self.noise_dim])
        seed_labels = [j for i in range(6) for j in range(self.label_dim)]
        seed_labels = to_categorical(seed_labels)
        return y0, y1, seed_noises, seed_labels
    
    def _make_randoms(self, batch_size):
        random_noises = tf.random.normal(shape=[batch_size, self.noise_dim])
        random_labels = np.random.randint(0, self.label_dim, batch_size).reshape(-1,1)
        random_labels = to_categorical(random_labels, self.label_dim)
        return random_noises, random_labels 

    def train(self, x_data, y_data, epochs, batch_size, save_path=None):

        ## set data ##
        dataset = self._make_datasets(x_data, y_data, batch_size)
        
        ## make constant ##
        # y1 : all value '0' and shape is (batch_size, )
        # y2 : all value '1' and shape is (bathc_size, )
        # seed_noise, seed_label : random values and shape is ( batch_size * z_dimension )
        y0, y1, seed_noises, seed_labels = self._make_constants(batch_size)

        ## train-epoch  ##
        history = {'d_loss':[], 'g_loss':[]}
        for epoch in range(1, epochs+1):
            print("Epoch {}/{}".format(epoch, epochs))
            
            ## train-batch ##
            d_loss, g_loss = 0, 0
            for real_imgs, labels in dataset:

                # phase 1 - train discriminator
                random_noises, _ = self._make_randoms(batch_size) 
                fake_imgs = self.gene.predict([random_noises, labels])
                
                self.disc.trainable = True
                d_loss_real = self.disc.train_on_batch([real_imgs, labels], y1)
                d_loss_fake = self.disc.train_on_batch([fake_imgs, labels], y0)
                d_loss = d_loss + (0.5*d_loss_real) + (0.5*d_loss_fake)

                # phase 2 - train generator
                randome_noises, random_labels = self._make_randoms(batch_size) 
                
                self.disc.trainable = False
                g_loss = g_loss + self.cgan.train_on_batch([random_noises, random_labels], y1)
            
            ## save results ##
            print('d_loss:%f, g_loss:%f'%(d_loss, g_loss))
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)

            if save_path: save_name = '%s/smaple_%i'%(save_path, epoch)
            else: save_name = None        
            self.plot_sample_images( seed_noises, seed_labels, save_name=save_name)

        return history
        

    def plot_sample_images(self, noises, labels, save_name=None):
        r, c = 6, self.label_dim
        gen_imgs = self.gene.predict([noises, labels])

        #Rescale images 0 - 1
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(c,r))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(np.squeeze(gen_imgs[cnt, :,:,:]), cmap = 'gray_r')
                axs[i,j].axis('off')
                cnt += 1

        if save_name==None:
            plt.show()
        else:
            fig.savefig(save_name)
            plt.close()

    def plot_model(self, save_path):
        plot_model(self.cgan, to_file='%s/cgan.png'%save_path, show_shapes=True)
        plot_model(self.gene, to_file='%s/gene.png'%save_path, show_shapes=True)
        plot_model(self.disc, to_file='%s/disc.png'%save_path, show_shapes=True)

    def save_model(self, save_path):
        self.cgan.save('%s/cgan.h5'%save_path)
        self.gene.save('%s/gene.h5'%save_path)
        self.disc.save('%s/disc.h5'%save_path)

    def load_weights(self, file_path):
       self.cgan.load_weights(file_path)
