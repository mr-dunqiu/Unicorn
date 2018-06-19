import os
import sys


class Config():
    instance = None

    def __init__(self):
        # self.currenpath = sys.path[0]

        # wu
        # self.currenpath = "/home/imc/caffe/poster/0406/tensorflow"
        self.currenpath = "e:/caffe_poster_0406_tensorflow"

        # self.library_path = "/home/imc/caffe/poster/0406/library"
        # wu
        # self.library_path = "/home/imc/models-master/inception/mydata/postertest1"
        self.library_path = "e:/models-master/inception/mydata/postertest1"

        self.hashfeature_path = self.currenpath + "/hashfeature.bin"
        self.feature_path = self.currenpath + "/feature.bin"
        self.groundtruth_path = self.currenpath + "/groundtruth.txt"
        self.FEATURE_LENGTH = 2048
        self.HASH_LENGTH = 128

        self.caffe_root = '/home/imc/caffe/'
        self.net_file = self.caffe_root + 'poster/models/bvlc_alexnet/deploy_hash.prototxt'
        self.caffe_model = self.caffe_root + 'poster/models/bvlc_alexnet/caffe_alexnet_train_iter_826.caffemodel'
        self.mean_file = self.caffe_root + 'poster/data/all/mean.npy'

        self.khash = 5

    def Getlibrary_path(self):
        return self.library_path

    def Getgroundtruth_path(self):
        return self.groundtruth_path

    def Gethashfeature_path(self):
        return self.hashfeature_path

    def Getfeature_path(self):
        return self.feature_path

    def GetFEATURE_LENGTH(self):
        return self.FEATURE_LENGTH

    def GetHASH_LENGTH(self):
        return self.HASH_LENGTH

    def Getcaffe_root(self):
        return self.caffe_root

    def Getnet_file(self):
        return self.net_file

    def Getcaffe_model(self):
        return self.caffe_model

    def Getmean_file(self):
        return self.mean_file

    def Getkhash(self):
        return self.khash


a = Config();

