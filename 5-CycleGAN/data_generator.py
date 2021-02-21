import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

'''
apple2orrange
summer2winter_yosemite
horse2zebra
monet2photo
cezanne2photo
ukiyoe2photo
vangogh2photo
maps
cityscapes
facades
ipohne2dslr_flower
'''

class DataGenerator():
    def __init__(self, data_name, width, height):

        self.data_name = data_name
        self.width = width
        self.height = height
        
        self.dataset, self.metadata = tfds.load('cycle_gan/%s'%data_name,
                              with_info=True, as_supervised=True)


    def _img_process(self, data, mode='train'):
        data_out = []
        for d in data:
            if mode== 'train':
                jittersize=30
            else: 
                jittersize=0

            img = tf.image.resize(d[0], [self.height+jittersize, self.width+jittersize], 
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            if mode == 'train':
                # randomly cropping
                img = tf.image.random_crop(img, size=[self.height, self.width, 3])
                # random mirroring
                img = tf.image.random_flip_left_right(img)

            img = np.array(img)
            img = (img/127.5)-1
            data_out.append(img)

        return  np.array(data_out)

    def get_dataset(self):
        a_train, b_train = self.dataset['trainA'], self.dataset['trainB']
        a_train = self._img_process(a_train)
        b_train = self._img_process(b_train)
        
        a_test, b_test= self.dataset['testA'], self.dataset['testB']
        a_test = self._img_process(a_test, mode='test')
        b_test = self._img_process(b_test, mode='test')

        mn = min(a_train.shape[0], b_train.shape[0])
        a_train, b_train = a_train[:mn], b_train[:mn]
        mn = min(a_test.shape[0], b_test.shape[0])
        a_test, b_test = a_test[:mn], b_test[:mn]


        return a_train, b_train, a_test, b_test

def main():

    DG = DataGenerator('horse2zebra', 128, 128)
    a_train, b_train, a_test, b_test = DG.get_dataset()
    
    print(a_train.shape, b_train.shape, a_test.shape, b_test.shape)    

if __name__=='__main__':
    main()
