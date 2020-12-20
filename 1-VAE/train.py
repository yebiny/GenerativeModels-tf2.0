import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import plot_model
from models import *
from drawTools import *

class VAE():

    def __init__(self, x_train, x_valid, save_path, z_dim=100):
        
        self.x_train = x_train
        self.x_valid = x_valid
        self.save_path = save_path
        self.z_dim = z_dim

        self.vae, self.encoder, self.decoder = build_vae(x_data.shape, self.z_dim)
        plot_model(self.encoder, to_file=save_path+'/encoder.png', show_shapes=True)
        plot_model(self.decoder, to_file=save_path+'/decoder.png',show_shapes=True)
        plot_model(self.vae, to_file=save_path+'/vae.png',show_shapes=True)

    def make_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.x_data)).shuffle(1)
        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(1)
        return dataset
    
    def get_rec_loss(self, inputs, predictions):
        rec_loss = tf.keras.losses.binary_crossentropy(inputs, predictions)
        rec_loss = tf.reduce_mean(rec_loss)
        rec_loss *= self.x_data.shape[1]*self.x_data.shape[2]
        return rec_loss
    
    def get_kl_loss(self, z_log_var, z_mean):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss
 

    def train(self, n_epochs):

        # set data
        dataset = self.make_datasets()
        
        # compile..? 
        optimizer = tf.keras.optimizers.Adam(0.001)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        
        #plot_rec_images(self.vae, self.x_data[:100], save=self.save_path+'recImg_0')
        
        # Start epoch loop
        history= {'epoch':[], 't_loss':[]}
        for epoch in range(1, n_epochs+1):
            for inputs, outputs in dataset:
                with tf.GradientTape() as tape:
                    # Get model ouputs
                    z_log_var, z_mean, z = self.encoder(inputs)
                    predictions = self.decoder(z)
                    # Compute losses
                    rec_loss = self.get_rec_loss(inputs, predictions)
                    kl_loss = self.get_kl_loss(z_log_var, z_mean)
                    loss = rec_loss + kl_loss
            
                # Compute gradients
                varialbes = self.vae.trainable_variables
                gradients = tape.gradient(loss, varialbes)
                # Update weights
                optimizer.apply_gradients(zip(gradients, varialbes))
            
                # Update train loss
                train_loss(loss)
            
            # Get loss and leraning rate at this epoch
            t_loss = train_loss.result().numpy() 
        
            # Save loss, lerning rate
            print("* %i * loss: %f "%(epoch, t_loss))
            history['epcoh'].append(epoch)
            history['t_loss'].append(t_loss)
            plot_reconstructed_images(self.vae, self.x_data[:100], x_save=self.save_path+'recImg_i'%epoch)
            
            # Reset loss
            train_loss.reset_states()   
            valid_loss.reset_states()
   
        return history
