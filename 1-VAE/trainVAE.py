import tensorflow as tf
import numpy as np
import pandas as pd

from models import *
from basic import *

class TrainVAE():

    def __init__(self, x_data, y_data, latent_dim,  save_path,  ckp='y'):
        self.x_data, self.y_data = x_data, y_data
       
        encoder, decoder, vae = build_vae(x_data.shape[1:], latent_dim)
        self.encoder = encoder
        self.decoder = decoder
        self.vae = vae
       
        self.save_path = save_path
        self.ckp_dir = self.save_path+'/ckp/'

        self.checkpoint = tf.train.Checkpoint(step=tf.Variable(1), encoder=self.encoder, decoder=self.decoder, vae=self.vae)
        if ckp=='y':
            self.checkpoint.restore(tf.train.latest_checkpoint(self.ckp_dir))

    def get_rec_loss(self, inputs, predictions):
        rec_loss = tf.keras.losses.binary_crossentropy(inputs, predictions)
        rec_loss = tf.reduce_mean(rec_loss)
        rec_loss *= self.x_data.shape[1]*self.x_data.shape[2]
        return rec_loss
    
    def get_kl_loss(self, z_log_var, z_mean):
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss
  
    @tf.function
    def train_step(self, inputs, train_loss, optimizer):
        with tf.GradientTape() as tape:
    
            # Get model ouputs
            z_log_var, z_mean, z = self.encoder(inputs)
            predictions = self.decoder(z)
    
            # Compute losses
            rec_loss = self.get_rec_loss(inputs, predictions)
            kl_loss = self.get_kl_loss(z_log_var, z_mean)
            loss = rec_loss + kl_loss
    
        # Compute gradients
        varialbes = self.vae.trainable_variables
        gradients = tape.gradient(loss, varialbes)
        # Update weights
        optimizer.apply_gradients(zip(gradients, varialbes))
    
        # Update train loss
        train_loss(loss)
        
        
    @tf.function
    def valid_step(self, inputs, valid_loss):
        with tf.GradientTape() as tape:

            # Get model ouputs without training
            z_log_var, z_mean, z = self.encoder(inputs, training=False)
            predictions = self.decoder(z, training=False)

            # Compute losses
            rec_loss = self.get_rec_loss(inputs, predictions)
            kl_loss = self.get_kl_loss(z_log_var, z_mean)
            loss = rec_loss + kl_loss

        # Update valid loss 
        valid_loss(loss)


    def train(self, epochs, batch_size, init_lr=0.001):
     
        x_train, x_valid, _, _, _, _ = data_generator(self.x_data, self.y_data, batch_size)
        x_train, x_valid = x_train/255, x_valid/255
        
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, x_train)).batch(batch_size)
        valid_ds = tf.data.Dataset.from_tensor_slices((x_valid, x_valid)).batch(batch_size)

        csv_logger = tf.keras.callbacks.CSVLogger(self.save_path+'/training.log')
        optimizer = tf.keras.optimizers.Adam(init_lr)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        valid_loss = tf.keras.metrics.Mean(name='valid_loss') 
        
        # Initialize values
        best_loss, count = float('inf'), 0
        
        # Start epoch loop
        for epoch in range(epochs):
            
            for inputs, outputs in train_ds:
                self.train_step(inputs, train_loss, optimizer)
            
            for inputs, outputs in valid_ds:
                self.valid_step(inputs, valid_loss)
            
            # Get loss and leraning rate at this epoch
            t_loss = train_loss.result().numpy() 
            v_loss = valid_loss.result().numpy()
            l_rate = optimizer.learning_rate.numpy()
        
            # Control learning rate
            count, lr  = reduce_lr(best_loss, v_loss, count, l_rate, 5, 0.2, 0.00001)
            optimizer.learning_rate = lr
            
            # Plot reconstruct image per 10 epochs
            draw_recimg(self.vae, x_valid, epoch, 5, self.save_path)
            
            # Save checkpoint if best v_loss 
            if v_loss < best_loss:
                best_loss = v_loss
                self.checkpoint.save(file_prefix=os.path.join(self.save_path+'/ckp/', 'ckp'))
            
            # Save loss, lerning rate
            print("* %i * loss: %f, v_loss: %f,  best_loss: %f, l_rate: %f, lr_count: %i"%(epoch, t_loss, v_loss, best_loss, l_rate, count ))
            df = pd.DataFrame({'epoch':[epoch], 'loss':[t_loss], 'v_loss':[v_loss], 'best_loss':[best_loss], 'l_rate':[l_rate]  } )
            df.to_csv(self.save_path+'/process.csv', mode='a', header=False)
            
    
            # Reset loss
            train_loss.reset_states()   
            valid_loss.reset_states()
            
def main():
    opt = argparse.ArgumentParser()
    opt.add_argument('-d',  dest='data_path', type=str, required=True, help='datasets path')
    opt.add_argument('-s',  dest='save_path', type=str, required=True, help='save path')
    opt.add_argument('-z',  dest='z_dim', type=int, default=100, required=False, help='latent space dimension')
    opt.add_argument('-e',  dest='epohs', type=int, default=3000, required=False, help='epochs')
    opt.add_argument('-b',  dest='batch_size', type=int, default=256, required=False, help='batch size')
    argv = opt.parse_args()

    if_exist(argv.save_path)
    if_not_make(argv.save_path)
    
    print('* Save at ', argv.save_path)
    print('* LAT_DIM, EPOCHS, BATCH_SIZE: ', argv.z_dim, argv.epochs, argv.batch_size)
    
    log  = '''
    Dataset : {DATASET} 
    Save path : {SAVE_PATH}
    Latent dimension : {LAT_DIM}
    Epochs : {EPOCHS}
    Batch size : {BATCH_SIZE}
    '''.format(DATASET=argv.data_path, SAVE_PATH=argv.save_path, LAT_DIM=argv.z_dim, EPOCHS=argv.epochs, BATCH_SIZE=argv.batch_size)
    with open(SAVE_PATH+'/log.txt', 'a') as log_file:
        log_file.write(log)
    with open(SAVE_PATH+'/log.txt', 'w') as log_file:
        log_file.write(log)
    

    x_data, y_data = np.load('%s/x_data.npy'%argv.data_path), np.load('%s/y_data.npy'%argv.data_path)
    tVAE = TrainVAE(x_data, y_data, argv.z_dim, argv.save_path)
    tVAE.vae.summary()
    tVAE.train(argv.epochs, argv.batch_size)

if __name__ == '__main__':
    main()
