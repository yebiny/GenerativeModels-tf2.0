import numpy as np
from buildModel import *
from dataGenerator import *
from drawTools import *
from tensorflow.keras.utils import plot_model

class TrainCycleGAN():
    def __init__(self, dataset, save_path, img_shape=(128,128,3), batch_size=1):
        
        self.dataset = dataset
        self.img_shape = img_shape
        self.batch_size = batch_size
        self.disc_path = (8, 8, 1)
        self.save_path = save_path 
        
        CycleGAN = BuildCycleGAN(img_shape)
        self.gene_ab, self.gene_ba, self.disc_a, self.disc_b, self.cyclegan = CycleGAN.build_cyclegan()
        self.cyclegan.summary()
        
        plot_model(self.gene_ab, to_file=save_path+'/gene.png', show_shapes=True)        
        plot_model(self.disc_a, to_file=save_path+'/disc.png', show_shapes=True)        
        plot_model(self.cyclegan, to_file=save_path+'/cyclegan.png', show_shapes=True)        
    
    def make_datasets(self, buffer_size=1000):
        loader = DataLoader(self.dataset, self.img_shape, self.batch_size, buffer_size)
        train_a, train_b, test_a, test_b = loader.generate()
        return train_a, train_b, test_a, test_b

    def make_constant(self):
        y1 = np.ones((self.batch_size, ) + self.disc_path)
        y0 = np.zeros((self.batch_size, ) + self.disc_path)
        return y0, y1

    def compile(self, optimizer=tf.optimizers.Adam(0.0002, 0.5), lambda_cycle=10.0):
        lambda_id = 0.9*lambda_cycle
        self.cyclegan.compile(loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[1,1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
                              optimizer=optimizer)        
        
    def train(self, epochs):
        y0, y1 = self.make_constant()
        train_a, train_b, test_a, test_b = self.make_datasets()
        sample_a, sample_b = next(iter(train_a)), next(iter(train_b))

        for epoch in range(epochs):
            for batch_idx, (imgs_a, imgs_b) in enumerate(tf.data.Dataset.zip(
                                                  (train_a, train_b))):
                fake_b = self.gene_ab.predict(imgs_a)
                fake_a = self.gene_ba.predict(imgs_b)
                self.disc_a.trainable = True
                self.disc_b.trainable = True
                
                da_loss_real = self.disc_a.train_on_batch(imgs_a, y1)
                da_loss_fake = self.disc_a.train_on_batch(fake_a, y0)
                da_loss = 0.5*np.add(da_loss_real, da_loss_fake)
                
                db_loss_real = self.disc_b.train_on_batch(imgs_b, y1)
                db_loss_fake = self.disc_b.train_on_batch(fake_b, y0)
                db_loss = 0.5*np.add(db_loss_real, db_loss_fake)
                
                d_loss = 0.5 *np.add(da_loss, db_loss)

                g_loss = self.cyclegan.train_on_batch([imgs_a, imgs_b],
                                                     [y1, y1, imgs_a, imgs_b, imgs_a, imgs_b]) 
                            
                if batch_idx%100==0: 
                    generate_img(self.gene_ab, self.gene_ba, sample_a, sample_b, 
                                 self.save_path+'/geneImg_%i_%i'%(epoch, batch_idx) )        
                    print('* %i - %i  :'%(epoch, batch_idx), da_loss, db_loss, g_loss)

            self.disc_a.save(self.save_path+'/disc_a_%i.h5'%epoch)  
            self.disc_b.save(self.save_path+'/disc_b_%i.h5'%epoch)  
            self.gene_ab.save(self.save_path+'/gene_ab_%i.h5'%epoch)  
            self.gene_ba.save(self.save_path+'/gene_ba_%i.h5'%epoch)  
