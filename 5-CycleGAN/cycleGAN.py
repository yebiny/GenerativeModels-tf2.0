import numpy as np
from buildModel import *
from dataGenerator import *
from drawTools import *
from tensorflow.keras.utils import plot_model

class CycleGAN():
    def __init__(self, input_shape, gene_n_filters, disc_n_filters, save_path):
        
        # set init vars
        self.input_shape = input_shape
        patch=int(self.input_shape[0]/2**4)
        self.disc_patch = (patch, patch, 1)
        self.save_path = save_path 
       
        # build cyclegan
        builder = BuildCycleGAN(input_shape, gene_n_filters, disc_n_filters)
        self.gene_ab, self.gene_ba, self.disc_a, self.disc_b, self.cyclegan = builder.build_cyclegan()
        self.cyclegan.summary()
        
        # draw model images
        plot_model(self.gene_ab, to_file=save_path+'/gene.png', show_shapes=True)        
        plot_model(self.disc_a, to_file=save_path+'/disc.png', show_shapes=True)        
        plot_model(self.cyclegan, to_file=save_path+'/cyclegan.png', show_shapes=True)        
    

    def compile(self, optimizer, lambda_valid, lambda_cycle, lambda_ident):
        
        self.disc_a.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.disc_b.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
        self.disc_a.trainable=False
        self.disc_b.trainable=False
        self.cyclegan.compile(loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[lambda_valid, lambda_valid, 
                                            lambda_cycle, lambda_cycle, 
                                            lambda_ident, lambda_ident],
                              optimizer=optimizer)        
    
    def make_constant(self, batch_size):
        y1 = np.ones((batch_size, ) + self.disc_patch)
        y0 = np.zeros((batch_size, ) + self.disc_patch)
        return y0, y1
        
    def train(self, data_loader, epochs, batch_size):
        
        # set data
        train_a, train_b, test_a, test_b = data_loader.generate(batch_size)
        sample_a, sample_b = next(iter(train_a)), next(iter(train_b))
        y0, y1 = self.make_constant(batch_size)
        
        # train
        history = {'epoch':[], 'd_loss':[], 'g_loss':[]}
        generate_img(self.gene_ab, self.gene_ba, sample_a, sample_b, self.save_path+'/geneImg_0' )        
        for epoch in range(1, epochs+1):
            
            gene_loss=disc_loss=0
            print("* epoch {}/{}".format(epoch, epochs))
            for batch_idx, (imgs_a, imgs_b) in enumerate(tf.data.Dataset.zip((train_a, train_b))):
            
                fake_b = self.gene_ab.predict(imgs_a)
                fake_a = self.gene_ba.predict(imgs_b)
                
                self.disc_a.trainable = True
                da_loss_real = self.disc_a.train_on_batch(imgs_a, y1)
                da_loss_fake = self.disc_a.train_on_batch(fake_a, y0)
                da_loss = 0.5*np.add(da_loss_real, da_loss_fake)

                self.disc_b.trainable = True
                db_loss_real = self.disc_b.train_on_batch(imgs_b, y1)
                db_loss_fake = self.disc_b.train_on_batch(fake_b, y0)
                db_loss = 0.5*np.add(db_loss_real, db_loss_fake)
                
                d_loss = 0.5*np.add(da_loss, db_loss)
                
                self.disc_a.trainable = False
                self.disc_b.trainable = False
                g_loss = self.cyclegan.train_on_batch([imgs_a, imgs_b],
                                                     [y1, y1, imgs_a, imgs_b, imgs_a, imgs_b]) 
                
                disc_loss = disc_loss + d_loss[0]
                gene_loss = gene_loss + g_loss[0]

            print('datasize: ', batch_idx*batch_size)
            print('* d_loss: ', disc_loss, 'g_loss: ' , gene_loss)
            history['epoch'].append(epoch)
            history['d_loss'].append(disc_loss)
            history['g_loss'].append(gene_loss)
            generate_img(self.gene_ab, self.gene_ba, sample_a, sample_b, 
                         self.save_path+'/geneImg_%i_%i'%(epoch, batch_idx) )        

            self.save()
        
        return history
    
    def save(self):
        self.disc_a.save(self.save_path+'/disc_a.h5')  
        self.disc_b.save(self.save_path+'/disc_b.h5')  
        self.gene_ab.save(self.save_path+'/gene_ab.h5')  
        self.gene_ba.save(self.save_path+'/gene_ba.h5') 
