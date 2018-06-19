'''
'''
# coding = 'utf-8'

import tensorflow as tf
import numpy as np
from src.inception import inception_module

class FeatureExtracter():
    def __init__(self,config):
        self.__config=config
        self.x_image = None
        self.__image_buffer = None
        self.logits = None
        self.highfeatures = None
        self.ssss = None
        self.__output = None
        self.saver = None
        self.__sess = None
        self.__sess = None
        self.__buildNet()
        self.__loadModel()

    def __loadModel(self):
        self.saver.restore(self.__sess, self.__config['checkpoint_path'])

    def __buildInputImagePlaceholder(self):
        self.__image_buffer = tf.placeholder("string")
        image = tf.image.decode_jpeg(self.__image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.central_crop(image, central_fraction=0.875)
        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [299, 299], align_corners=False)
        image_tensor = tf.squeeze(image, [0])
        self.x_image = tf.reshape(image_tensor, [-1, 299, 299, 3])

    def __buildNet(self):
        #graph = tf.Graph().as_default()
        # Number of classes in the dataset label set plus 1.
        # Label 0 is reserved for an (unused) background class.
        num_classes = self.__config['num_classes']+1
        # setup an input image placeholder to feed image buffer
        self.__buildInputImagePlaceholder()
        # Build a Graph that computes the logits predictions from the inference model.
        # WARNING!!!
        self.logits,self.highfeatures, self.ssss = inception_module.inference(self.x_image, num_classes)
        # result is the output of the softmax unit
        self.__output = tf.nn.softmax(self.ssss, name="result")
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            inception_module.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        self.saver = tf.train.Saver(variables_to_restore)
        self.__sess = tf.Session()

    def extract(self, image_path):
        '''提取图像特征

        参数
        image_path: 图像检索路径
        '''
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        output, hashFeature, imageFeature = self.__sess.run([self.__output, self.logits, self.highfeatures],
                                                  feed_dict={self.__image_buffer: image_data})
        hashFeature = np.squeeze(hashFeature)
        imageFeature = np.squeeze(imageFeature[0])
        imageFeature = np.array([imageFeature])
        hashFeature = np.array([hashFeature])
        for i in range(len(hashFeature[0])):
            if hashFeature[0][i] > 0.5:
                hashFeature[0][i] = 1
            else:
                hashFeature[0][i] = 0
        return hashFeature, imageFeature

if __name__ == '__main__':
    pass