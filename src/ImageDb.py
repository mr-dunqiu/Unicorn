'''
'''
# coding = 'utf-8'

import cv2
import numpy as np


class ImageDb(object):
    '''图像库
    '''
    _FLANN_INDEX_KDTREE = 1
    _FLANN_INDEX_PARAM = {'algorithm': _FLANN_INDEX_KDTREE, 'trees': 4}
    _K_OF_KNN = 1
    _HASH_FEATURE_DIM = 128
    _IMGE_FEATURE_DIM = 2048


    def __init__(self, config):
        '''
        参数
        config: 配置
        '''

        self.__config = config
        self.__hashFeatures = np.fromfile(config['hashFeatureFilename'], dtype=np.float32)
        self.__hashFeatures.shape = self.__hashFeatures.size // ImageDb._HASH_FEATURE_DIM, ImageDb._HASH_FEATURE_DIM
        self.__imageFeatures = np.fromfile(config['imageFeatureFilename'],dtype=np.float32)
        self.__imageFeatures.shape = self.__imageFeatures.size // ImageDb._IMGE_FEATURE_DIM, ImageDb._IMGE_FEATURE_DIM
        # self.__imageLabels = np.fromfile(config['imageLabelsFilename'],dtype=np.string_)
        imageLabels = np.loadtxt(config['labelsFilename'],dtype=np.string_).astype(str).tolist()
        self.__imageLabels = imageLabels if isinstance(imageLabels, list) else [imageLabels]
        self.__indexer = cv2.flann_Index(self.__hashFeatures, ImageDb._FLANN_INDEX_PARAM)


    def find(self, hashFeature, imageFeature):
        '''查找图像Label

        参数
        hashFeature: 图像的Hash特征
        imageFeature: 图像特征
        '''
        candidates = self.__findCandidate(hashFeature)
        refinedCandidates = self.__refineCandidate(candidates, imageFeature)
        label = [self.__imageLabels[refinedCandidate] for refinedCandidate in refinedCandidates]
        print(label)
        return label


    def __findCandidate(self, hashFeature):
        _seedCandidates, _ = self.__indexer.knnSearch(hashFeature, ImageDb._K_OF_KNN, params={})
        seedCandidates = _seedCandidates[0].tolist()
        candidates = list(seedCandidates)
        for i in seedCandidates:
            #all(1)输出一个bool型矩阵，每个元素指示原矩阵一行中所有元素是否都是true
            auxCandidates = np.where((self.__hashFeatures == self.__hashFeatures[i]).all(1))[0]
            candidates.extend(auxCandidates)
        return list(set(candidates))


    def __refineCandidate(self, candidates, imageFeature):
        distWithQuery = lambda index: np.linalg.norm(self.__imageFeatures[index] - imageFeature[0])
        refinedCandidates = min(candidates, key=distWithQuery)
        return [refinedCandidates]


if __name__ == '__main__':
    pass
