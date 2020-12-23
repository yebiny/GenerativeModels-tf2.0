import os, sys
import argparse
from cycleGAN import *
 
def parse_args():
    opt = argparse.ArgumentParser(description="==== GAN with tensorflow2.x ====")
    opt.add_argument(dest='save_path', type=str, help=': set save directory ')
    opt.add_argument('--data', type=str, default='apple2orange', 
    help='choice among [apple2orange, summer2winter_yosemite, horse2zebra, monet2photo,cezanne2photo, ukiyoe2photo, vangogh2photo,maps, cityscapes, facades, ipohne2dslr_flower]')
    
    opt.add_argument('-e',  dest='epochs', type=int, default=5, help=': number epochs (default: 5)')
    opt.add_argument('-b', dest='batch_size', type=int, default=32, help=': batch size (default: 32)')
    args = opt.parse_args()
    return args

def check_args(args):
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
def main():
   
    IMG_SHAPE=(128,128,3)
    BUFFER_SIZE=1000

    args=parse_args()
    check_args(args)
    
    # make dataset
    data_loader = DataLoader(data_name=args.data, 
                             img_shape=IMG_SHAPE, 
                             buffer_size=BUFFER_SIZE)
    # Make model
    model = CycleGAN(input_shape=IMG_SHAPE, 
                     gene_n_filters=32,
                     disc_n_filters=32,
                     save_path=args.save_path)
    # Compile model
    model.compile(optimizer=tf.optimizers.Adam(0.0002, 0.5), 
                  lambda_valid=1, 
                  lambda_cycle=10, 
                  lambda_ident=9)
  
    # Train model
    history = model.train(data_loader, epochs=args.epochs, batch_size=args.batch_size)
    
    # Plot loss
    plot_loss(history, args.save_path+'/loss.png')

if __name__=='__main__':
    main()

