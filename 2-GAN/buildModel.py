import tensorflow as tf
from tensorflow.keras import layers, models

def build_gan(data_shape, noise_dim):
    w = data_shape[1] 
    h = data_shape[2]
    d = data_shape[3]
   
    inputs_noise = layers.Input(shape=[noise_dim,])
    
    y = layers.Dense(int(w/4) * int(h/4) * 128)(inputs_noise)
    y = layers.Reshape([int(w/4), int(h/4), 128])(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(d, kernel_size=5, strides=2, padding="SAME", activation="tanh")(y)
    generator=models.Model(inputs_noise, y, name='Generator')

    inputs_image = layers.Input(shape=[w,h,d])
    y = layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME", activation=layers.LeakyReLU(0.2))(inputs_image)
    y = layers.Dropout(0.4)(y)
    y = layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME", activation=layers.LeakyReLU(0.2))(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Flatten()(y)
    y = layers.Dense(1, activation="sigmoid")(y)
    discriminator=models.Model(inputs_image, y, name='Discriminator')
   
    gan = models.Sequential([generator, discriminator], name='CGAN')
    
    return gan

def main():
    data_shape=(1, 36,36,1)
    noise_dim=100
    gan  = build_gan(data_shape, noise_dim)
    generator, discriminator=gan.layers
    
    generator.summary()
    discriminator.summary()
    gan.summary()

if __name__=='__main__':
    main()
