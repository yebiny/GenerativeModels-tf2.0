from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pickle

class WGAN():
    def __init__(self, generator, critic):

        self.name = 'wgan'
        self.generator = generator
        self.critic = critic
        self.z_dim = generator.input_shape[1]
        
    def compile( self
               , optimizer
               , generator_lr = 0.00005
               , critic_lr = 0.00005):
        
        def _get_opti(lr):
            if optimizer == 'adam':
                opti = optimizers.Adam(lr=lr, beta_1=0.5)
            elif optimizer == 'rmsprop':
                opti = optimizers.RMSprop(lr=lr)
            else:
                opti = optimizers.Adam(lr=lr)
            return opti
        
        def _wasserstein(y_true, y_pred):
            return - K.mean(y_true * y_pred)
        
        def _set_trainable(m, val):
            m.trainable = val
            for l in m.layers:
                l.trainable = val
        
        ### COMPILE critic
        self.critic.compile( optimizer = _get_opti(critic_lr)
                             , loss = _wasserstein )
        
        ### COMPILE THE FULL GAN
        _set_trainable(self.critic, False)

        model_input = layers.Input(shape=self.z_dim, name='model_input')
        model_output = self.critic(self.generator(model_input))
        self.model = models.Model(model_input, model_output)
        
        self.model.compile(
            optimizer=_get_opti(generator_lr)
            , loss=_wasserstein )

        _set_trainable(self.critic, True)
   

    def _train_critic(self, x_train, batch_size, clip_threshold, using_generator):

        valid = np.ones((batch_size,1))
        fake = -np.ones((batch_size,1))
        
        # true images
        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
       
        # generated images
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        
        # train critic
        d_loss_real = self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)
        
        return [d_loss, d_loss_real, d_loss_fake]
        
    def _train_generator(self, batch_size):
        
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        g_loss =  self.model.train_on_batch(noise, valid)
        return g_loss

    def train( self, x_train, batch_size, epochs
             , save_path = None
             , save_iter = 10
             , n_critic = 5
             , clip_threshold = 0.01
             , using_generator = False):

        history={'d_loss':[], 'g_loss':[]}
          
        for epoch in range(1, epochs+1):
            
            # Train step
            for _ in range(n_critic):
                d_loss = self._train_critic(x_train, batch_size, clip_threshold, using_generator)
            g_loss = self._train_generator(batch_size)
               
            
            # Plot & Save the progress
            print ("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, d_loss[0], d_loss[1], d_loss[2], g_loss))
            history['d_loss'].append(d_loss[0])
            history['g_loss'].append(g_loss)
            
            if save_path:
                if epoch%save_iter==0: self.plot_sample_images('%s/sample_%i'%(save_path, epoch))
        
        if save_path:
            self.save_model(save_path)
            self.plot_model(save_path)
            
            with open('%s/history.pickle'%save_path,'wb') as fw:
                pickle.dump(history, fw)
            
        
        return history
    
        
    def plot_sample_images(self, save_name=None):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        #Rescale images 0 - 1
        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15,15))
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
        plot_model(self.model, to_file='%s/model.png'%save_path, show_shapes = True, show_layer_names = True)
        plot_model(self.critic, to_file='%s/critic.png'%save_path, show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file='%s/generator.png'%save_path, show_shapes = True, show_layer_names = True)
    
    def save_model(self, save_path):
        self.model.save('%s/model.h5'%save_path)
        self.critic.save('%s/critic.h5'%save_path)
        self.generator.save('%s/generator.h5'%save_path)
        #pickle.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
