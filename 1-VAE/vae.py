import tensorflow as tf
import os, sys
import numpy as np
from functools import reduce
from build_models import *
import matplotlib.pyplot as plt


class TrainVAE():

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
        self.img_shape = encoder.input_shape[1:]

    def compile(self, optimizer = tf.keras.optimizers.Adam(0.001)):
        inputs = layers.Input(shape=self.img_shape, name = 'vae_inputs')
        z_log_var, z_mean, z = self.encoder(inputs)
        outputs = self.decoder(z)
        self.vae = models.Model(inputs, outputs, name = 'vae')
        self.optimizer = optimizer
    
    def _make_dataset(self, x_train, batch_size):
        train_ds = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
        return train_ds
    
    def _get_rec_loss(self, inputs, predictions):
        rec_loss = tf.keras.losses.binary_crossentropy(inputs, predictions)
        rec_loss = tf.reduce_mean(rec_loss)
        rec_loss *= self.img_shape[0]*self.img_shape[1]
        return rec_loss
    
    def _get_kl_loss(self, z_log_var, z_mean):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss
    
    @tf.function
    def _train_step(self, imgs):
        with tf.GradientTape() as tape:
    
            # Get model ouputs
            z_log_var, z_mean, z = self.encoder(imgs)
            rec_imgs = self.decoder(z)
    
            # Compute losses
            rec_loss = self._get_rec_loss(imgs, rec_imgs)
            kl_loss = self._get_kl_loss(z_log_var, z_mean)
            loss = rec_loss + kl_loss
    
        # Compute gradients
        varialbes = self.vae.trainable_variables
        gradients = tape.gradient(loss, varialbes)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, varialbes))
    
        return loss, rec_loss, kl_loss 

    def train(self, x_train, epochs=1, batch_size=16):
        
        train_ds = self._make_dataset(x_train, batch_size)
        history = {'loss':[], 'rec_loss':[], 'kl_loss':[]} 
        
        for epoch in range(1, 1+epochs):
            for key in history: history[key].append(0)
            
            for batch_imgs in train_ds:
                losses = self._train_step(batch_imgs)
                for key, loss in zip(history, losses): 
                    history[key][-1]+=loss
             
            print('* epoch: %i, loss: %f, rec_loss: %f, kl_loss: %f'%( epoch
                                                                     , history['loss'][-1]
                                                                     , history['rec_loss'][-1]
                                                                     , history['kl_loss'][-1])) 

            self.plot_sample_imgs(batch_imgs)
        
        return history 

    def plot_sample_imgs(self, imgs, n=10, save_path=None):
        plt.figure(figsize=(n,2))
        rec_imgs = self.vae.predict(imgs[:n])
        for i, (img, rec_img) in enumerate(zip(imgs, rec_imgs)):
            plt.subplot(2,n,i+1)
            plt.imshow(np.squeeze(img), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
            plt.subplot(2,n,n+i+1)
            plt.imshow(np.squeeze(rec_img), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
        
        if save_path!=None: 
            plt.savefig(save_path)
        else: plt.show()

    def save_model(self, save_path):
        self.encoder.save('%s/encoder.h5'%save_path)
        self.decoder.save('%s/decoder.h5'%save_path)
        self.vae.save('%s/vae.h5'%save_path)
