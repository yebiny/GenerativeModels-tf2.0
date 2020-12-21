import numpy as np
import matplotlib.pyplot as plt

def generate_img(model_ab, model_ba, input_a, input_b, save=None):
    a2b = model_ab.predict(input_a)
    b2a = model_ba.predict(input_b)
    
    plt.figure(figsize=(8, 8))
    display_list = [input_a[0], a2b[0], input_b[0], b2a[0]]
    title = ['Input A', 'A to B', 'Input B', 'B to A']

    for idx, img in enumerate(display_list):
        plt.subplot(2, 2, idx+1)
        plt.title(title[idx])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(img * 0.5 + 0.5)
        plt.axis('off')

    if save==None:
        plt.show()
    else:
        plt.savefig(save)

    plt.close('all')
