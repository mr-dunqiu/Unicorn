'''
'''
# coding = 'utf-8'

import ImageDb
import FeatureExtract
import numpy as np

class ImageFinder(object):
    def __init__(self,config):
        self.__imageDb = ImageDb.ImageDb(config['db'])
        self.__featureExtracter = FeatureExtract.FeatureExtracter(config['extracter'])


    def find(self, imagePath):
        '''查找图像的Label

        参数
        imagePath: 图像文件路径
        '''
        hashFeature, imageFeature = self.__featureExtracter.extract(imagePath)
        # hashFeature = np.zeros(shape=[1, 128], dtype='float32')
        # imageFeature = np.zeros(shape=[1, 2048], dtype='float32')
        return self.__imageDb.find(hashFeature, imageFeature)



if __name__ == '__main__':

    config = {
        'db': {
            'hashFeatureFilename': r'D:\dev\unicorn\test\data\__NImage\feature\1.bin',
            'imageFeatureFilename': r'D:\dev\unicorn\test\data\__NImage\feature\2.bin',
            'generatedLabelsFilename': r'D:\dev\unicorn\test\data\__NImage\feature\generatedLabels.txt',
        },
        'extracter': {
            'num_classes': 11,
            'checkpoint_path': 'D:/dev/unicorn/model/model.ckpt-42100'
        }
    }
    imageFinder = ImageFinder(config)
    imagePath = r'D:\dev\unicorn\test\data\__NImage\image\0\0.jpg '
    imageFinder.find(imagePath)

