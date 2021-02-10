import os, sys
import argparse
sys.path.append('../')
from preprocess import *
from train import *

def parse_args():

    opt = argparse.ArgumentParser()
    opt.add_argument('-d',  dest='data', type=str, required=True, help='datasets')
    opt.add_argument('-s',  dest='save_path', type=str, required=True, help='save path')
    opt.add_argument('-z',  dest='z_dim', type=int, default=100, required=False, help='latent space dimension')
    opt.add_argument('-e',  dest='epochs', type=int, default=3000, required=False, help='epochs')
    opt.add_argument('-b',  dest='batch_size', type=int, default=256, required=False, help='batch size')
    args = opt.parse_args()
    
    return args

def check_args(args):
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    if args.data =='mnist':
        x_train, _ = get_mnist()
    elif args.data =='fmnist':
        x_train, _ = get_fashion_mnist()
    elif args.data =='cifar':
        x_train, _ = get_cifar10()

    return x_train


def main():
    
    args=parse_args()
    x_data = check_args(args)

    print('* Save at ', args.save_path)
    print('* LAT_DIM, EPOCHS, BATCH_SIZE: ', args.z_dim, args.epochs, args.batch_size)
    
    log  = '''
    Dataset : {DATASET} 
    Save path : {SAVE_PATH}
    Latent dimension : {LAT_DIM}
    Epochs : {EPOCHS}
    Batch size : {BATCH_SIZE}
    '''.format(DATASET=args.data, SAVE_PATH=args.save_path, LAT_DIM=args.z_dim, EPOCHS=args.epochs, BATCH_SIZE=args.batch_size)
    with open(args.save_path+'/log.txt', 'a') as log_file:
        log_file.write(log)
    with open(args.save_path+'/log.txt', 'w') as log_file:
        log_file.write(log)
    

    x_train, x_valid = train_test_split(x_data, test_size=0.3)
    print(x_train.shape, x_valid.shape )
    model = TrainVAE(x_train, x_valid, args.save_path, args.z_dim)
    model.vae.summary()
    model.train(args.epochs, args.batch_size)

if __name__ == '__main__':
    main()
