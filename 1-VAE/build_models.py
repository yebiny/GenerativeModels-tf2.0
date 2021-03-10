import argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

class BuildModel():
    def __init__(self, img_shape, z_dim, n_conv_layers=3, dense_dim=256):
        self.img_shape = img_shape
        self.z_dim = z_dim
        self.n_conv_layers = n_conv_layers
        self.dense_dim = dense_dim

    def _sampling(self, args):
        """Reparameterization function by sampling from an isotropic unit Gaussian.
        # Arguments:
            args (tensor): mean and log of variance of Q(z|X)
        # Returns:
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    
    def build_encoder(self):
        inputs = layers.Input(shape=self.img_shape)
        
        y = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
        y = layers.LeakyReLU()(y)
        for n in range(self.n_conv_layers-1):
            y = layers.Conv2D(32, 3, strides=2, padding="same")(y)
            y = layers.LeakyReLU()(y)
        self.y_shape = y.shape
        
        y = layers.Flatten()(y)
        y = layers.Dense(self.dense_dim, activation='relu')(y)
        y = layers.Dense(self.dense_dim, activation='relu')(y)
        
        z_mean = layers.Dense(self.z_dim, name="z_mean")(y)
        z_log_var = layers.Dense(self.z_dim, name="z_log_var")(y)
        z = layers.Lambda(self._sampling)([z_mean, z_log_var])
        
        encoder = models.Model(inputs, [z_mean, z_log_var, z], name ='encoder')
        
        return encoder
    
    def build_decoder(self):
        decoder_input = layers.Input(shape=(self.z_dim,))
    
        y = layers.Dense(self.dense_dim, activation='relu')(decoder_input)
        y = layers.Dense(self.dense_dim, activation='relu')(y)
        y = layers.Dense( self.y_shape[1]*self.y_shape[2]*self.y_shape[3]
                        , activation="relu")(y)
        y = layers.Reshape(self.y_shape[1:])(y)
       
        for n in range(self.n_conv_layers-1):
            y = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(y)
            y = layers.LeakyReLU()(y)
        y = layers.Conv2DTranspose(self.img_shape[-1], 3, strides=2, padding="same")(y)
        y = layers.Activation('sigmoid')(y)
    
        decoder = models.Model(decoder_input, y, name='decoder')
    
        return decoder

def main():
    
    builder = BuildModel((28,28,1), 100)
    encoder = builder.build_encoder()
    decoder = builder.build_decoder()
    
    encoder.summary()
    decoder.summary()

if __name__ == '__main__':
    main()
