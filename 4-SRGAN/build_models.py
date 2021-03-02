import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16

def build_generator( input_shape
                   , n_ResidualBlocks=16
                   , momentum=0.8):
    
    def _ResidualBlock(x):

        y = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(x)
        y = layers.Activation(activation="relu")(y)
        y = layers.BatchNormalization(momentum=momentum)(y)
        
        y = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")(y)
        y = layers.Activation(activation="relu")(y)
        y = layers.BatchNormalization(momentum=momentum)(y)
  
        y = layers.Add()([y, x])
        
        return y 
    
    def _ConvLayer(x, filters, kernel_size, activation=None):
        y = layers.Conv2D( filters=filters
                         , kernel_size=kernel_size
                         , strides=1
                         , padding='same')(x)
        if activation:
            y = layers.Activation(activation)(y)

        return y
    
    input_layer = layers.Input(shape = input_shape)
    x = _ConvLayer(input_layer, filters=64, kernel_size=9, activation='relu')

    y = _ResidualBlock(x)
    for n in range(n_ResidualBlocks-1): y = _ResidualBlock(y)
    
    y = _ConvLayer(y, filters=64, kernel_size=3)
    y = layers.BatchNormalization(momentum=momentum)(y)
    y = layers.Add()([y, x])

    y = layers.UpSampling2D(size=2)(y)
    y = _ConvLayer(y, filters=256, kernel_size=3, activation='relu')

    y = layers.UpSampling2D(size=2)(y)
    y = _ConvLayer(y, filters=256, kernel_size=3, activation='relu')

    output = _ConvLayer(y, filters=3, kernel_size=9, activation = 'tanh')
    
    return models.Model(inputs=[input_layer], outputs=[output], name='G')

def build_discriminator( input_shape
                       , leakyrelu_alpha=0.2
                       , bn_momentum = 0.8):

    def _ConvBlock(x, filters, strides, bn=False):
        y = layers.Conv2D( filters=filters, strides=strides 
                         , kernel_size=3, padding='same')(x)
        y = layers.LeakyReLU(alpha=leakyrelu_alpha)(y)
        if bn==True: y = layers.BatchNormalization(momentum=bn_momentum)(y)
        return y

    input_layer = layers.Input(shape=input_shape)
    
    y = _ConvBlock(input_layer, filters=64, strides=1)
    y = _ConvBlock(y, filters = 64, strides=2, bn=True)
    y = _ConvBlock(y, filters = 128, strides=1, bn=True)
    y = _ConvBlock(y, filters = 128, strides=2, bn=True)
    y = _ConvBlock(y, filters = 256, strides=1, bn=True)
    y = _ConvBlock(y, filters = 256, strides=2, bn=True)
    y = _ConvBlock(y, filters = 512, strides=1, bn=True)
    y = _ConvBlock(y, filters = 512, strides=2, bn=True)
    y = layers.Dense(units=1024)(y)
    y = layers.LeakyReLU(alpha=0.2)(y)
    
    output = layers.Dense(units=1, activation='sigmoid')(y)

    return models.Model(inputs=[input_layer], outputs=[output], name='D')

def build_vgg(input_shape):
   
    
    input_layer = layers.Input(shape=input_shape)
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor = input_layer)
    vgg_output = vgg(input_layer)
    y = layers.Flatten()(vgg_output)
    y = layers.Dense(4096)(y)
    output = layers.Dense(1000, activation='relu')(y) 
    return models.Model(inputs=[input_layer], outputs=[output], name='VGG')

