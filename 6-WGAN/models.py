import numpy as np
import os

from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import layers, models

def build_generator( input_shape
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
    generator_input = layers.Input(shape=input_shape, name='generator_input')

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
                , critic_conv_filters = [32,64,128,128]
                , critic_conv_kernel_size = [5,5,5,5]
                , critic_conv_strides = [2,2,2,1]
                , critic_batch_norm_momentum = None
                , critic_activation = 'leaky_relu'
                , critic_dropout_rate = None
                , critic_weight_init= RandomNormal(mean=0., stddev=0.02)
                ):

        critic_n_layers=len(critic_conv_filters)
        critic_input = layers.Input(shape=input_shape, name='critic_input')

        x = critic_input
        for i in range(critic_n_layers):
            
            x = layers.Conv2D(
                filters = critic_conv_filters[i]
                , kernel_size = critic_conv_kernel_size[i]
                , strides = critic_conv_strides[i]
                , padding = 'same'
                , name = 'critic_conv_' + str(i)
                , kernel_initializer = critic_weight_init
                )(x)
            
            if critic_batch_norm_momentum and i > 0:
                x = layers.BatchNormalization(momentum = critic_batch_norm_momentum)(x)

            x = layers.LeakyReLU(alpha = 0.2)(x)

            if critic_dropout_rate:
                x = layers.Dropout(rate = critic_dropout_rate)(x)

        x = layers.Flatten()(x)

        critic_output = layers.Dense(1, activation=None
                        , kernel_initializer = critic_weight_init)(x)

        return models.Model(critic_input, critic_output)


