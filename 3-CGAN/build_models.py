import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models



def build_generator(img_shape, label_dim, noise_dim):
    w, h, d = img_shape[0], img_shape[1], img_shape[2] 

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

    return generator

def build_discriminator(img_shape, label_dim):
    w, h, d = img_shape[0], img_shape[1], img_shape[2] 

    # Discriminator 
    def expand_label_input(x):
           x = K.expand_dims(x,axis=1)
           x = K.expand_dims(x,axis=1)
           x = K.tile(x, [1,int(w/2),int(h/2),1])
           return x
    
    input_image = layers.Input(shape=[w,h,d], name='Input_image')
    conv_image = layers.Conv2D( 64, kernel_size=5, strides=2, padding="SAME"
                     , activation=layers.LeakyReLU(0.2))(input_image)
    input_label = layers.Input(shape=[label_dim,], name='Input_label')
    expand_label = layers.Lambda(expand_label_input)(input_label)
    
    
    y = layers.concatenate([conv_image, expand_label], axis=3) 
    y = layers.Conv2D( 128, kernel_size=5, strides=2, padding="SAME"
                     , activation=layers.LeakyReLU(0.2))(y)
    y = layers.Dropout(0.4)(y)
    y = layers.Flatten()(y)
    decision = layers.Dense(1, activation="sigmoid", name='Decision')(y)
    
    discriminator=models.Model(inputs=[input_image, input_label], outputs=[decision], name='Discriminator')

    return discriminator

def main():
    
    IMG_SHAPE=(28, 28, 1)
    LABEL_DIM=10
    NOISE_DIM=100
    
    gene = build_generator( IMG_SHAPE, LABEL_DIM, NOISE_DIM )
    disc = build_discriminator( IMG_SHAPE, LABEL_DIM )

    
    gene.summary()
    discsummary()

if __name__=='__main__':
    main()
