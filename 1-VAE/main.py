import os, sys
import argparse
from getData import *
from train import *

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
    opt = argparse.ArgumentParser()
    opt.add_argument('-d',  dest='data', type=str, required=True, help='datasets')
    opt.add_argument('-s',  dest='save_path', type=str, required=True, help='save path')
    opt.add_argument('-z',  dest='z_dim', type=int, default=100, required=False, help='latent space dimension')
    opt.add_argument('-e',  dest='epochs', type=int, default=3000, required=False, help='epochs')
    opt.add_argument('-b',  dest='batch_size', type=int, default=256, required=False, help='batch size')
    argv = opt.parse_args()

    print('* Save at ', argv.save_path)
    print('* LAT_DIM, EPOCHS, BATCH_SIZE: ', argv.z_dim, argv.epochs, argv.batch_size)
    
    log  = '''
    Dataset : {DATASET} 
    Save path : {SAVE_PATH}
    Latent dimension : {LAT_DIM}
    Epochs : {EPOCHS}
    Batch size : {BATCH_SIZE}
    '''.format(DATASET=argv.data, SAVE_PATH=argv.save_path, LAT_DIM=argv.z_dim, EPOCHS=argv.epochs, BATCH_SIZE=argv.batch_size)
    with open(argv.save_path+'/log.txt', 'a') as log_file:
        log_file.write(log)
    with open(argv.save_path+'/log.txt', 'w') as log_file:
        log_file.write(log)
    

    #x_data, y_data = np.load('%s/x_data.npy'%argv.data_path), np.load('%s/y_data.npy'%argv.data_path)
    x_data = check_args(argv)
    x_train, x_valid = train_test_split(x_data, test_size=0.3)
    print(x_train[0][:20] )
    tVAE = TrainVAE(x_train, x_valid, argv.save_path, argv.z_dim)
    tVAE.vae.summary()
    tVAE.train(argv.epochs, argv.batch_size)

if __name__ == '__main__':
    main()
