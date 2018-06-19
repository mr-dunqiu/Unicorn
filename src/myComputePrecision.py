from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
import sys, os
import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_module
FLAGS = tf.app.flags.FLAGS

#checkpoint_path=E:\老电脑\实验室\海报检索1\海报检索\inceptionmodel.ckpt-42100
#/home/imc/caffe/poster/mytensor0417/model11/model.ckpt-42100
tf.app.flags.DEFINE_string('checkpoint_path', 'D:/dev/unicorn/model/model.ckpt-42100',
                           """Directory where to read model checkpoints.""")
#num_classes=11,训练数据的类别数
tf.app.flags.DEFINE_string('num_classes', 11,
                           """class numbers.""")


class NetSaver():
  def __init__(self):
    self.loadNet()
    self.loadModel()

  def loadNet(self):
    self.__buildNet()

  def loadModel(self, model=FLAGS.checkpoint_path):
    self.saver.restore(self.__sess, model)

#@property将私有成员变成具有只读属性的公有成员
  @property
  def classes(self):
    return self.__classes

  @property
  def sess(self):
    return self.__sess

  @property
  def output(self):
    return self.__output

  @property
  def image_buffer(self):
    return self.__image_buffer

  def __load_model(self):
    #self.__sess = tf.Session()
    self.saver.restore(self.__sess, self.model)

  def __buildInputImagePlaceholder(self):
    self.__image_buffer = tf.placeholder("string")
    image = tf.image.decode_jpeg(self.image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.central_crop(image, central_fraction=0.875)
    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [299, 299],align_corners=False)
    image_tensor = tf.squeeze(image, [0])
    self.x_image = tf.reshape(image_tensor, [-1, 299, 299, 3])

  def __buildNet(self):
    graph = tf.Graph().as_default()
    # Number of classes in the dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = FLAGS.num_classes + 1
    #setup an input image placeholder to feed image buffer
    self.__buildInputImagePlaceholder()
    # Build a Graph that computes the logits predictions from the inference model.
    self.logits,self.highfeatures, self.ssss = inception_module.inference(self.x_image, num_classes)
    #result is the output of the softmax unit
    self.__output = tf.nn.softmax(self.ssss, name="result")
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception_module.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    self.saver = tf.train.Saver(variables_to_restore)
    self.__sess = tf.Session()

  def classify(self, image_path):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # output:分到每一类的概率
    output,predictions,high  = self.sess.run([self.output,self.logits,self.highfeatures],feed_dict={self.image_buffer: image_data})
    index=0
    pro=0
    for i in range(len(output[0])):
      if output[0][i]>pro:
        pro=output[0][i]
        index=i
    return index

  # def predict(self):
  #   img1 = "E:/老电脑/实验室/海报检索1/测试视频/10.jpg"
  #   feat1 = self.getOneFeatures(img1)
  #   ss=self.classify(img1)
  #   #print(len(feat1))
  #   img2 = "/home/imc/caffe/data/package/500-999/highlight_resize150/614.2.jpg"
  #   feat2 = self.getOneFeatures(img2)
  #   #print(len(feat2))def computeprecision(self):

  # def computeprecision(self):
  #   roots=[]
  #   imgsroot0='D:/dev/unicorn/data/testVideo/video0/'
  #   imgsroot1='D:/dev/unicorn/data/testVideo/video1/'
  #   imgsroot2='D:/dev/unicorn/data/testVideo/video2/'
  #   #roots.append(imgsroot0)
  #   roots.append(imgsroot1)
  #   #roots.append(imgsroot2)
  #   count=0
  #   lab=1
  #   for imgroot in roots:
  #     for root, dirs, files in os.walk(imgroot):
  #         for i in range(len(files)):
  #           img=imgroot+files[i]
  #           index=self.classify(img)
  #           if(index!=2):
  #             count=count+1
  #             #print (index)
  #     lab=lab+1

if __name__ == '__main__':
    a=NetSaver()
    print(a.classify('D:/dev/unicorn/data/testVideo/10.jpg'))
