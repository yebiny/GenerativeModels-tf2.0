import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models


def build_cgan(x_shape, y_shape, noise_dim):
    w = x_shape[1] 
    h = x_shape[2]
    d = x_shape[3]
    label_dim = y_shape[-1]

    # Generator
    input_noise = layers.Input(shape=[noise_dim,], name='Input_noise')
    input_label = layers.Input(shape=[label_dim,], name='Input_label')
    
    y = layers.concatenate([input_noise, input_label])
    y = layers.Dense(int(w/4) * int(h/4) * 128)(y)
    y = layers.Reshape([int(w/4), int(h/4), 128])(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu")(y)
    y = layers.BatchNormalization()(y)
    generated_image = layers.Conv2DTranspose(d, kernel_size=5, strides=2, padding="SAME", activation="tanh", name='generated_image')(y)
    
    generator=models.Model(inputs=[input_noise, input_label], outputs=[generated_image], name='Generator')
    
    # Discriminator 
    def expand_label_input(x):
           x = K.expand_dims(x,axis=1)
           x = K.expand_dims(x,axis=1)
           x = K.tile(x, [1,int(w/2),int(h/2),1])
           return x
    
    input_image = layers.Input(shape=[w,h,d], name='Input_image')
    expand_label = layers.Lambda(expand_label_input)(input_label)
    
    y = layers.Conv2D(64, kernel_size=5, strides=2, padding="SAME", activation=layers.LeakyReLU(0.2))(input_image)
    y = layers.concatenate([y, expand_label], axis=3) 
    y = layers.Conv2D(128, kernel_size=5, strides=2, padding="SAME", activation=layers.LeakyReLU(0.2))(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Flatten()(y)
    decision = layers.Dense(1, activation="sigmoid", name='Decision')(y)
    
    discriminator=models.Model(inputs=[input_image, input_label], outputs=decision, name='Discriminator')

    generated_images=generator([input_noise, input_label])
    decision = discriminator([generated_images, input_label])
    cgan = models.Model(inputs=[input_noise, input_label], outputs=[decision], name='CGAN')

    return cgan, generator, discriminator

def main():
    x_shape=(1, 28, 28,1)
    y_shape=(1,10)
    noise_dim=100

    cgan, generator, discriminator  = build_cgan(x_shape, y_shape, noise_dim)
    
    generator.summary()
    discriminator.summary()
    cgan.summary()

if __name__=='__main__':
    main()
