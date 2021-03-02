import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, models

class BuildModels():
    def __init__(self, img_shape, noise_dim, label_dim):
        self.img_shape = img_shape
        self.noise_dim = noise_dim
        self.label_dim = label_dim

    def build_generator(self):
        w, h, d = self.img_shape
    
        input_noise = layers.Input(shape=[self.noise_dim,], name='Input_noise')
        input_label = layers.Input(shape=[self.label_dim,], name='Input_label')
        y = layers.concatenate([input_noise, input_label])
        y = layers.Dense(int(w/4) * int(h/4) * 128)(y)
        y = layers.Reshape([int(w/4), int(h/4), 128])(y)
        y = layers.BatchNormalization()(y)
        y = layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="SAME", activation="selu")(y)
        y = layers.BatchNormalization()(y)
        output = layers.Conv2DTranspose(d, kernel_size=5, strides=2, padding="SAME", activation="tanh")(y)
        
        generator=models.Model(inputs=[input_noise, input_label], outputs=[output], name='G')
    
        return generator

    def build_discriminator(self):
        w, h, d = self.img_shape
    
        def _expand_label_input(x):
               x = K.expand_dims(x,axis=1)
               x = K.expand_dims(x,axis=1)
               x = K.tile(x, [1,int(w/2),int(h/2),1])
               return x
        
        input_image  = layers.Input(shape=[w,h,d], name='Input_image')
        y_img = layers.Conv2D( 64, kernel_size=5, strides=2, padding="same")(input_image)
        y_img = layers.LeakyReLU(0.2)(y_img)

        input_label = layers.Input(shape=[self.label_dim,], name='Input_label')
        y_label = layers.Lambda(_expand_label_input)(input_label)
        
        y = layers.concatenate([y_img, y_label], axis=3) 
        y = layers.Conv2D(128, kernel_size=5, strides=2, padding="same")(y)
        y = layers.LeakyReLU(0.2)(y)

        y = layers.Dropout(0.4)(y)
        y = layers.Flatten()(y)
        output = layers.Dense(1, activation="sigmoid", name='Decision')(y)
        
        discriminator=models.Model( inputs = [input_image, input_label]
                                  , outputs= [output]
                                  , name='D')
    
        return discriminator

def main():
    
    IMG_SHAPE=(28, 28, 1)
    LABEL_DIM=10
    NOISE_DIM=100
    
    builder = BuildModels(IMG_SHAPE, NOISE_DIM, LABEL_DIM)
    gene = builder.build_generator()
    disc = builder.build_discriminator()

    gene.summary()
    disc.summary()

if __name__=='__main__':
    main()
