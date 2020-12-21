import numpy as np
from buildModel import *
from drawTools import *
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.optimizers import Adam

class CGAN():
    def __init__(self, x_data, y_data, save_path, noise_dim=100, batch_size=32):
    
        self.x_data = x_data
        self.y_data = y_data
        self.save_path = save_path
        self.noise_dim = noise_dim
        self.label_dim = y_data.shape[-1]
        self.batch_size= batch_size
        
        self.cgan, self.generator, self.discriminator = build_cgan(x_data.shape, y_data.shape, noise_dim)
        plot_model(self.cgan, to_file=save_path+'/cgan.png', show_shapes=True)
        plot_model(self.generator, to_file=save_path+'/gene.png',show_shapes=True)
        plot_model(self.discriminator, to_file=save_path+'/disc.png',show_shapes=True)
    
    def compile(self, optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)):
        # When compile generator(gan), discriminator must not trainable!
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer = optimizer)
        self.discriminator.trainable = False
        self.cgan.compile(loss="binary_crossentropy", optimizer=optimizer)

    def make_datasets(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.x_data, self.y_data)).shuffle(1)
        dataset = dataset.batch(self.batch_size, drop_remainder=True).prefetch(1)
        return dataset

    def make_constants(self):
        y0 = tf.constant([[0.]] * self.batch_size)
        y1 = tf.constant([[1.]] * self.batch_size)
        seed_noises = tf.random.normal(shape=[6*self.label_dim, self.noise_dim])
        seed_labels = [j for i in range(6) for j in range(self.label_dim)]
        seed_labels = to_categorical(seed_labels)
        return y0, y1, seed_noises, seed_labels
    
    def make_randoms(self):
        random_noises = tf.random.normal(shape=[self.batch_size, self.noise_dim])
        random_labels = np.random.randint(0, self.label_dim, self.batch_size).reshape(-1,1)
        random_labels = to_categorical(random_labels, self.label_dim)
        return random_noises, random_labels 

    def train(self, n_epochs):

        # set data
        dataset = self.make_datasets()

        # y1 : all value '0' and shape is (batch_size, )
        # y2 : all value '1' and shape is (bathc_size, )
        # seed_noise, seed_label : random values and shape is ( batch_size * z_dimension )
        y0, y1, seed_noises, seed_labels = self.make_constants()
        plot_generated_images(self.generator, seed_noises, seed_labels, save=self.save_path+'/generatedImg_0')

        # train
        history = {'epoch':[], 'd_loss':[], 'g_loss':[]}
        for epoch in range(1, n_epochs+1):
            print("Epoch {}/{}".format(epoch, n_epochs))
            
            d_loss=g_loss=0
            for x_real, labels in dataset:
                # phase 1 - train discriminator
                random_noises, _ = self.make_randoms() 
                x_fake = self.generator.predict([random_noises, labels])

                self.discriminator.trainable = True
                dl1 = self.discriminator.train_on_batch([x_real, labels], y1)
                dl2 = self.discriminator.train_on_batch([x_fake, labels], y0)
                d_loss = d_loss + (0.5*dl1) + (0.5*dl2)

                # phase 2 - train generator
                randome_noises, random_labels = self.make_randoms() 
                
                self.discriminator.trainable = False
                gl = self.cgan.train_on_batch([random_noises, random_labels], y1)
                g_loss = g_loss + gl
            
            print('d_loss:%f, g_loss:%f'%(d_loss, g_loss))
            history['epoch'].append(epoch)
            history['d_loss'].append(d_loss)
            history['g_loss'].append(g_loss)
            plot_generated_images(self.generator, seed_noises, seed_labels, save=self.save_path+'/generatedImg_%i'%epoch)
            
        return history
    
