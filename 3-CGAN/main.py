import os, sys
import argparse
from getData import *
from trainCGAN import *

def parse_args():
    opt = argparse.ArgumentParser(description="==== GAN with tensorflow2.x ====")
    opt.add_argument(dest='save_path', type=str, help=': set save directory ')
    opt.add_argument('--data', type=str, default='fmnist', help=': choice among [mnist / fmnist / cifar] (default: [fmnist] )')
    opt.add_argument('-e',  dest='epochs', type=int, default=5, help=': number epochs (default: 5)')
    
    args = opt.parse_args()

    return args

def check_args(args):
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    
    if args.data =='mnist':
        x_train, y_train = get_mnist()
    elif args.data =='fmnist':
        x_train, y_train = get_fashion_mnist()
    elif args.data =='cifar':
        x_train, y_train = get_cifar10()

    return x_train, y_train


def main():
    
    args=parse_args()
    
    # Load dataset
    x_train, y_train = check_args(args)
    print('* Use dataset', args.data, x_train.shape, y_train.shape)
    
    # Make model
    cgan = CGAN(x_train, y_train, args.save_path)
    # Compile model
    cgan.compile()
    # Train model
    history = cgan.train(args.epochs)
    # Plot loss
    plot_loss(history, args.save_path+'/loss.png')

if __name__=='__main__':
    main()

