import os, sys
import argparse
from train import *
 
def parse_args():
    opt = argparse.ArgumentParser(description="==== GAN with tensorflow2.x ====")
    opt.add_argument(dest='save_path', type=str, help=': set save directory ')
    opt.add_argument('--data', type=str, default='apple2orange', 
                        help='choice among apple2orange, summer2winter_yosemite, horse2zebra, monet2photo')
    
    opt.add_argument('-e',  dest='epochs', type=int, default=5, help=': number epochs (default: 5)')
    
    args = opt.parse_args()
    return args

def check_args(args):
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
def main():
    
    args=parse_args()
    check_args(args)
    
    # Make model
    cycle_gan = TrainCycleGAN(args.data, args.save_path)
    # Compile model
    cycle_gan.compile()
    # Train model
    cycle_gan.train(args.epochs)
    # Plot loss
    #plot_loss(history, args.save_path+'/loss.png')

if __name__=='__main__':
    main()

