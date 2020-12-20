
# To plot pretty figures
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

def plot_multiple_images(images, n_cols=None, save=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
    if save!=None:
        plt.savefig(save)        
    else:
        plt.show()

def plot_generated_images(generator, seed, save=None):
    generated_images = generator.predict(seed)
    generated_images = (generated_images+1)/2
    plot_multiple_images(generated_images, 8, save) 

def plot_rec_images(reconstructor, data, save=None):
    reconstructed_images = reconstructor.predict(data)
    reconstructed_images = (reconstructed_images+1)/2
    plot_multiple_images(reconstructed_images, 8, save)

def plot_loss(history, save=None):
    figure=plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(history['epoch'])
    plt.plot(history['g_loss'], 'y', label='generator loss')
    plt.plot(history['d_loss'], 'r', label='discriminator loss')
    plt.legend(loc='upper right')

    if save!=None:
        plt.savefig(save)        
    else:
        plt.show()
