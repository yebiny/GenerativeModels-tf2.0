from tensorflow.keras import backend as K
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import os, sys

class WGAN():
    def __init__(self, generator, critic, run_folder):

        self.name = 'wgan'
        self.generator = generator
        self.critic = critic
        self.run_folder=run_folder
        if not os.path.exists(run_folder):
            os.mkdir(run_folder)
            os.mkdir(os.path.join(run_folder, 'viz'))
            os.mkdir(os.path.join(run_folder, 'images'))
            os.mkdir(os.path.join(run_folder, 'weights'))

        self.z_dim = generator.input_shape[1]
        
        self.d_losses = []
        self.g_losses = []
        self.epoch = 0
    
    def compile( self
               , optimizer
               , generator_lr
               , critic_lr):
        
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
                             , loss = _wasserstein
                           )
        
        ### COMPILE THE FULL GAN
        _set_trainable(self.critic, False)

        model_input = layers.Input(shape=self.z_dim, name='model_input')
        model_output = self.critic(self.generator(model_input))
        self.model = models.Model(model_input, model_output)
        
        self.model.compile(
            optimizer=_get_opti(generator_lr)
            , loss=_wasserstein
            )

        _set_trainable(self.critic, True)
   

    def train_critic(self, x_train, batch_size, clip_threshold, using_generator):

        valid = np.ones((batch_size,1))
        fake = -np.ones((batch_size,1))

        if using_generator:
            true_imgs = next(x_train)[0]
            if true_imgs.shape[0] != batch_size:
                true_imgs = next(x_train)[0]
        else:
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            true_imgs = x_train[idx]
        
        
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        gen_imgs = self.generator.predict(noise)

        d_loss_real =   self.critic.train_on_batch(true_imgs, valid)
        d_loss_fake =   self.critic.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        for l in self.critic.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)

        return [d_loss, d_loss_real, d_loss_fake]
    
    def train_generator(self, batch_size):
        valid = np.ones((batch_size,1))
        noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        return self.model.train_on_batch(noise, valid)

    def train( self, x_train, batch_size, epochs
             , print_every_n_batches = 10
             , n_critic = 5
             , clip_threshold = 0.01
             , using_generator = False):

          for epoch in range(self.epoch, self.epoch + epochs):

              for _ in range(n_critic):
                  d_loss = self.train_critic(x_train, batch_size, clip_threshold, using_generator)

              g_loss = self.train_generator(batch_size)
                 
              # Plot the progress
              print ("%d [D loss: (%.3f)(R %.3f, F %.3f)]  [G loss: %.3f] " % (epoch, d_loss[0], d_loss[1], d_loss[2], g_loss))
              
              self.d_losses.append(d_loss)
              self.g_losses.append(g_loss)

              # If at save interval => save generated image samples
              if epoch % print_every_n_batches == 0:
                  self.model.save_weights(os.path.join(self.run_folder, 'weights/weights-%d.h5' % (epoch)))
                  self.model.save_weights(os.path.join(self.run_folder, 'weights/weights.h5'))
                  self.save_model()
                  self.sample_images()
              
              self.epoch+=1


    def sample_images(self):
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
        fig.savefig(os.path.join(self.run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()

    def plot_model(self):
        plot_model(self.model, to_file=os.path.join(self.run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.critic, to_file=os.path.join(self.run_folder ,'viz/critic.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(self.run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)
    def save_model(self):
        self.model.save(os.path.join(self.run_folder, 'model.h5'))
        self.critic.save(os.path.join(self.run_folder, 'critic.h5'))
        self.generator.save(os.path.join(self.run_folder, 'generator.h5'))
        #pickle.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))
    
    def load_weights(self, filepath):
        self.model.load_weights(filepath)
