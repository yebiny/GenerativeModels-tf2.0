import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class SRGAN():
    def __init__( self, generator, discriminator, vgg, lr_shape, hr_shape):
    
        self.name = 'SRGAN'
        self.gene = generator
        self.disc = discriminator
        self.vgg = vgg
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
   
    def compile(self, optimizer=Adam( lr=0.0002
                                   , beta_1=0.5
                                   , beta_2=0.999
                                   , epsilon=10e-8)):
        
        self.vgg.trainable=False
        self.vgg.compile(loss='mse', optimizer=optimizer)
        self.disc.compile(loss='mse', optimizer=optimizer)

        lr_imgs = layers.Input(shape=self.lr_shape, name='LR')
        fake_hr_imgs = self.gene(lr_imgs)
        decision = self.disc(fake_hr_imgs)
        features = self.vgg(fake_hr_imgs)

        self.disc.trainable = False
        self.srgan = models.Model(inputs=[lr_imgs], outputs=[decision, features])
        self.srgan.compile( loss=['binary_crossentropy', 'mse']
                          , loss_weights=[1e-3, 1]
                          , optimizer=optimizer)
    
    def _make_datasets(self, lr_data, hr_data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((lr_data, hr_data)).shuffle(1)
        dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)
        return dataset

    def _make_constants(self, batch_size):
        
        shape=self.disc.output_shape
        shape=(batch_size, shape[1], shape[2], shape[3])
        y0 = np.zeros((shape))
        y1 = np.ones((shape))
        return y0, y1
    
    def train(self, lr_data, hr_data, epochs, batch_size, save_path=None):

        ## set data ##
        train_ds = self._make_datasets(lr_data, hr_data, batch_size)
        y0, y1 = self._make_constants(batch_size)

        ## epoch  ##
        history = {'d_loss':[], 'g_loss':[], 'vgg_loss':[]}
        for epoch in range(1, 1+epochs):
            for h in history: history[h].append(0)
            
            ## batch-trainset ##
            for lr_imgs, real_hr_imgs in train_ds:

                # phase 1 - train discriminator
                fake_hr_imgs = self.gene.predict_on_batch([lr_imgs])

                self.disc.trainable = True
                d_loss_real = self.disc.train_on_batch(real_hr_imgs, y1)
                d_loss_fake = self.disc.train_on_batch(fake_hr_imgs, y0)
                d_loss = (0.5*d_loss_real) + (0.5*d_loss_fake)

                # phase 2 - train generator
                self.disc.trainable = False
                real_features = self.vgg.predict_on_batch(real_hr_imgs)
                g_loss  = self.srgan.train_on_batch([lr_imgs], [y1, real_features])
                print(g_loss)
                history['d_loss'][-1]+=d_loss 
                history['g_loss'][-1]+=g_loss[0] 
            
            ## save results ##
            print('* epoch: %i, d_loss: %f, g_loss: %f'%( epoch
                                                        , history['d_loss'][-1]
                                                        , history['g_loss'][-1]))
            
            self.plot_sample_imgs(lr_imgs, real_hr_imgs)

        return history
        

    def plot_sample_imgs(self, lr_imgs, hr_imgs, save_name=None):
        row, col = 4, 9
        fake_hr_imgs = self.gene.predict([lr_imgs[:row*col]])

        #Rescale images 0 - 1
        lr_imgs = np.clip((0.5*(lr_imgs+1)), 0, 1)
        hr_imgs = np.clip((0.5*(hr_imgs+1)), 0, 1)
        fake_hr_imgs = np.clip((0.5*(fake_hr_imgs+1)), 0, 1)

        fig, axis = plt.subplots(row, col, figsize=(col*1.5,row*1.5))
        
        idx = 0
        for r in range(row):
            for c in range(col):
                if c%3==0: img = lr_imgs
                elif c%3==1: img = hr_imgs
                else: 
                    img = fake_hr_imgs
                    idx+=1
                axis[r,c].imshow(np.squeeze(img[idx, :,:,:]), cmap='gray_r')
                axis[r,c].axis('off')

        if save_name==None:
            plt.show()
        else:
            fig.savefig(save_name)
            plt.close()

    def plot_model(self, save_path):
        plot_model(self.srgan, to_file='%s/srgan.png'%save_path, show_shapes=True)
        plot_model(self.gene, to_file='%s/gene.png'%save_path, show_shapes=True)
        plot_model(self.disc, to_file='%s/disc.png'%save_path, show_shapes=True)

    def save_model(self, save_path):
        self.srgan.save('%s/srgan.h5'%save_path)
        self.gene.save('%s/gene.h5'%save_path)
        self.disc.save('%s/disc.h5'%save_path)

    def load_weights(self, file_path):
       self.srgan.load_weights(file_path)
