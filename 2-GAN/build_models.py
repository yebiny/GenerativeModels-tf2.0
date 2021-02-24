import tensorflow as tf
from tensorflow.keras import layers, models

class BuildModels():
    def __init__(self, data_shape, noise_dim):
        self.data_shape = data_shape
        self.noise_dim = noise_dim

    def build_generator(self):
        w, h, d = self.data_shape 
        
        # Generator
        inputs_noise = layers.Input(shape=[self.noise_dim,])
        y = layers.Dense(int(w/4) * int(h/4) * 128)(inputs_noise)
        y = layers.Reshape([int(w/4), int(h/4), 128])(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu")(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2DTranspose(d, kernel_size=5, strides=2, padding="SAME", activation="tanh")(y)
        generator=models.Model(inputs_noise, y, name='G')
        
        return generator

    def build_discriminator(self):
        inputs_image = layers.Input(shape=self.data_shape)
        y = layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME", activation=layers.LeakyReLU(0.2))(inputs_image)
        y = layers.Dropout(0.4)(y)
        y = layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME", activation=layers.LeakyReLU(0.2))(y)
        y = layers.Dropout(0.4)(y)
        y = layers.Flatten()(y)
        y = layers.Dense(1, activation="sigmoid")(y)
        discriminator=models.Model(inputs_image, y, name='D')
        
        return discriminator

def main():
    DATA_SHAPE= (28,28,1)
    NOISE_DIM = 100
    builder = BuildModels(DATA_SHAPE, NOISE_DIM)
    gene = builder.build_generator()
    disc = builder.build_discriminator()
    gene.summary()
    disc.summary()

if __name__=='__main__':
    main()
