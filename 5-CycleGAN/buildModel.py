import tensorflow as tf
tf.random.Generator = None
import tensorflow_addons as tfa
from tensorflow.keras import layers, activations, models

class BuildCycleGAN():
    
    def __init__(self, input_shape, gene_n_filters, disc_n_filters):
        self.input_shape = input_shape
        self.df = gene_n_filters
        self.gf = disc_n_filters
        
    def conv2d(self, layer_input, filters, f_size=4, nomalization=True):
        d = layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = layers.LeakyReLU(alpha=0.2)(d)
        if nomalization:
            d = tfa.layers.InstanceNormalization()(d)
        return d

    def deconv2d(self, layer_input, skip_input, filters, f_size=4):
        u = layers.UpSampling2D(size=2)(layer_input)
        u = layers.Conv2D(filters, kernel_size=f_size, strides=1, 
                          padding='same', activation='relu')(u)
        u = tfa.layers.InstanceNormalization()(u)
        u = layers.Concatenate()([u, skip_input])
        return u

    def build_generator(self, name='G'):

        d0 = layers.Input(shape=self.input_shape)

        d1 = self.conv2d(d0, self.df*1)
        d2 = self.conv2d(d1, self.df*2)
        d3 = self.conv2d(d2, self.df*4)
        d4 = self.conv2d(d3, self.df*8)

        u1 = self.deconv2d(d4, d3, self.gf*4)
        u2 = self.deconv2d(u1, d2, self.gf*2)
        u3 = self.deconv2d(u2, d1, self.gf*1)

        u4 = layers.UpSampling2D(size=2)(u3)
        output_img = layers.Conv2D(self.input_shape[2], kernel_size=4, strides=1,
                                  padding='same', activation='tanh')(u4)

        return models.Model(d0, output_img, name=name)

    def build_discriminator(self, name='D'):
        img = layers.Input(shape=self.input_shape)
        d1 = self.conv2d(img, self.df*1, nomalization=False)
        d2 = self.conv2d(d1, self.df*2)
        d3 = self.conv2d(d2, self.df*4)
        d4 = self.conv2d(d3, self.df*8)

        predict = layers.Conv2D(1, activation='sigmoid', 
                    kernel_size=4, strides=1, padding='same', name='test')(d4)
        return models.Model(img, predict, name=name)
    

def main():

    builder = BuildCycleGAN(input_shape=(128,128,3),
                            gene_n_filters=32,
                            disc_n_filters=32)
    gene = builder.build_generator()
    disc = builder.build_discriminator()
       
    gene.summary()
    disc.summary()
if __name__=='__main__':
    main()    
