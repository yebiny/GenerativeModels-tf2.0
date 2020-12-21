import tensorflow as tf
import tensorflow_datasets as tfds

class DataLoader():
    def __init__(self, dataset, img_shape = (128, 128, 3), batch_size = 1, buffer_size = 1000):
        self.buffer_size=buffer_size
        self.batch_size=batch_size
        self.img_width=img_shape[0]
        self.img_height=img_shape[1]
        
        dataset, metadata = tfds.load('cycle_gan/%s'%dataset,
                              with_info=True, as_supervised=True)

        self.train_a, self.train_b = dataset['trainA'], dataset['trainB']
        self.test_a, self.test_b = dataset['testA'], dataset['testB']
        
    def normalize(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def resize(self, image, height, width):
        # resizing
        image = tf.image.resize(image, [height, width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image

    def random_jitter(self, image):

        # randomly cropping
        image = tf.image.random_crop(image, size=[self.img_height, self.img_width, 3])

        # random mirroring
        image = tf.image.random_flip_left_right(image)

        return image

    def preprocess_image_train(self, image, label):
        image = self.resize(image, self.img_height+10, self.img_width+10)
        image = self.random_jitter(image)
        image = self.normalize(image)
        return image
    
    def preprocess_image_test(self, image, label):
        image = self.resize(image, self.img_height, self.img_width)
        image = self.normalize(image)
        return image
    
    def generate(self):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        train_a = self.train_a.map(
            self.preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
            self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(1)

        train_b = self.train_b.map(
            self.preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
            self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(1)

        test_a = self.test_a.map(
            self.preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(1)

        test_b = self.test_b.map(
            self.preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
            self.buffer_size).batch(self.batch_size, drop_remainder=True).prefetch(1)

        return train_a, train_b, test_a, test_b


def main():
    loader = DataLoader('apple2orange', (128, 128, 3), 1, 1000)
    train_a, train_b, test_a, test_b = loader.generate()
    print(train_a, train_b)    

if __name__=='__main__':
    main()
