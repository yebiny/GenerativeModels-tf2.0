import os, sys, glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd


def get_masked_label(labels, labels_val, vals):

    labels_masked=labels
    for val in vals:
        n = np.where(labels_val==val)[0][0]
        mask = (labels_masked[:,n]=='-1')
        labels_masked = labels_masked[mask]
    return labels_masked


def plot_imgs(labels, img_path, w=10, h=10, size=128):
    crop = (25, 55, 150, 185)
    fig = plt.figure(figsize=(w*2,h*2))

    for idx in range(w*h):
        name = labels[idx][0]
        image = Image.open(img_path+name)
        image = np.array((image.crop(crop)).resize((size,size)))

        plt.subplot(h,w, idx+1)
        plt.title(name)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
    plt.show()

def get_dataset_from_labels(labels, img_path, size=128):
    x_data=[]
    y_data=[]
    crop = (25, 55, 150, 185)
    idx = 0
    for label in labels:
        idx=idx+1
        name = label[0]
        image = Image.open(img_path+name)
        image = np.array((image.crop(crop)).resize((size,size)))

        if idx%500==0: print(idx)
        x_data.append(image)
        y_data.append(name)

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def main():
    csv_file = sys.argv[1]
    img_path = sys.argv[2]
    img_size = int(sys.argv[3])
    save_path = sys.argv[4]
    if not os.path.exists(save_path): os.makedirs(save_path)
    
    # READ CSV FILE
    csv_file = pd.read_csv(csv_file, sep = ",", dtype = 'unicode')
    labels = csv_file.values
    labels_val = csv_file.columns.values
    print( ' * lables: ', len(labels))
    print( ' * labels value: ', labels_val)
    
    masked_labels = get_masked_label(labels, labels_val, ['Blurry', 'Pale_Skin', 'Wearing_Hat', 'Double_Chin'])
    labels_used, _ = train_test_split(masked_labels,  train_size = 0.5, random_state=34)
    x_data, y_data = get_dataset_from_labels(labels_used, img_path, img_size)
    print( '* outputs:',  x_data.shape, y_data.shape)
    
    np.save('%s/x_data'%save_path, x_data)
    np.save('%s/y_data'%save_path, y_data)

if __name__ == '__main__':
    main()

