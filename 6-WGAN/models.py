import numpy as np
import os

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers, models

def build_generator( z_dim
                   , generator_initial_dense_layer_size = (4, 4, 128)
                   , generator_upsample = [2,2, 2,1]
                   , generator_conv_filters = [128,64,32,3]
                   , generator_conv_kernel_size = [5,5,5,5]
                   , generator_conv_strides = [1,1, 1,1]
                   , generator_batch_norm_momentum = 0.8
                   , generator_dropout_rate = None
                   , generator_weight_init = RandomNormal(mean=0., stddev=0.02)
                   ):

    generator_n_layers = len(generator_conv_filters)
    generator_input = layers.Input(shape=(z_dim), name='generator_input')

    x = generator_input
    x = layers.Dense(np.prod(generator_initial_dense_layer_size)
              ,kernel_initializer = generator_weight_init)(x)

    if generator_batch_norm_momentum:
        x = layers.BatchNormalization(momentum = generator_batch_norm_momentum)(x)

    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Reshape(generator_initial_dense_layer_size)(x)

    if generator_dropout_rate:
        x = layers.Dropout(rate = generator_dropout_rate)(x)

    for i in range(generator_n_layers):

        if generator_upsample[i] == 2:
            x = layers.UpSampling2D()(x)
            x = layers.Conv2D(
            filters = generator_conv_filters[i]
            , kernel_size = generator_conv_kernel_size[i]
            , padding = 'same'
            , name = 'generator_conv_' + str(i)
            , kernel_initializer = generator_weight_init
            )(x)
        else:

            x = layers.Conv2DTranspose(
                filters = generator_conv_filters[i]
                , kernel_size = generator_conv_kernel_size[i]
                , padding = 'same'
                , strides = generator_conv_strides[i]
                , name = 'generator_conv_' + str(i)
                , kernel_initializer = generator_weight_init
                )(x)

        if i < generator_n_layers - 1:
            if generator_batch_norm_momentum:
                x = layers.BatchNormalization(momentum = generator_batch_norm_momentum)(x)
            x = layers.LeakyReLU(alpha=0.2)(x)

        else:
            x = layers.Activation('tanh')(x)

    generator_output = x
    generator = models.Model(generator_input, generator_output)
    
    return generator



def build_critic( input_shape 
                , conv_filters = [32,64,128,128]
                , conv_strides = [2,2,2,1]
                , conv_kernel_size = [5,5,5,5]
                , weight_init= RandomNormal(mean=0., stddev=0.02)
                , bn_momentum = None
                , dropout_rate = None
                ):

        def _critic_conv(x, filters, kernel_size, strides, weight_init, bn_momentum, dropout_rate):
            
            x = layers.Conv2D( filters = filters
                             , strides = strides
                             , kernel_size = kernel_size
                             , kernel_initializer = weight_init
                             , padding = 'same'
                             )(x)
            if bn_momentum: x = layers.BatchNormalization(momentum = batch_norm_momentum)(x)
            x = layers.LeakyReLU(alpha = 0.2)(x)
            if dropout_rate: x = layers.Dropout(rate =dropout_rate)(x)

            return x

        critic_input = layers.Input(shape=input_shape, name='critic_input')
        
        x = critic_input
        for filters, kernel_size, strides in zip(conv_filters, conv_kernel_size, conv_strides):
            x = _critic_conv(x, filters, kernel_size, strides, weight_init, bn_momentum, dropout_rate)    

        x = layers.Flatten()(x)
        critic_output = layers.Dense(1, kernel_initializer=weight_init)(x)

        return models.Model(critic_input, critic_output)


