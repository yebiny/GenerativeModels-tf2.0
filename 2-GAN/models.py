import tensorflow as tf
from tensorflow.keras import layers, models

def build_gan(data_shape, noise_dim):
    w = data_shape[1] 
    h = data_shape[2]
    d = data_shape[3]
    
    generator = models.Sequential([
        layers.Dense(int(w/4) * int(h/4) * 128, input_shape=[noise_dim]),
        layers.Reshape([int(w/4), int(h/4), 128]),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME",
                                     activation="selu"),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(d, kernel_size=5, strides=2, padding="SAME",
                                     activation="tanh"),
    ])
    discriminator = models.Sequential([
        layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME",
                            activation=layers.LeakyReLU(0.2),
                            input_shape=[w, h, d]),
        layers.Dropout(0.4),
        layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME",
                            activation=layers.LeakyReLU(0.2)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    
    gan = models.Sequential([generator, discriminator])
    
    return gan
