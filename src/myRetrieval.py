import tensorflow as tf
import numpy as np
import cv2.flann
import heapq
# #import AA
# from AA import AA
# aa=AA.AA()

HASH_LENGTH=1
FEATURE_LENGTH=3
K=2

class Retrieval():
    #"""docstring fos Retrieval"""
    def __init__(self):
        self.buildHashFeatureIndex()
        self.buildEFeatureIndex()

    def buildHashFeatureIndex(self):
        pass

    def buildEFeatureIndex(self):
        pass

    def readImg(self,imgPath):
        image = tf.gfile.FastGFile(imgPath, 'rb').read()
        return image

    #提取待检索图片的特征
    def getFeatures(self,img):
        #hashfeature,feature,high = self.sess.run([self.output,self.logits,self.highfeatures],feed_dict={self.image_buffer: image_data})
        hashfeature = np.zeros(shape=[1, HASH_LENGTH], dtype='float32')
        feature = np.zeros(shape=[1, FEATURE_LENGTH], dtype='float32')
        #feature = np.squeeze(feature)
        #hashfeature = np.squeeze(high[0])
        # hashfeature = np.array([hashfeature])
        # feature = np.array([feature])
        # for i in range(len(feature[0])):
        #     if feature[0][i] > 0.5:
        #         feature[0][i] = 1
        #     else:
        #         feature[0][i] = 0
        return hashfeature, feature

    def search(self,feature1,feature2):
        candidate=self.findByHamming(feature1)
        label=self.findByE(candidate,feature2)
        return label

    def findByHamming(self,feature1):
        #kd-tree建索引
        FLANN_INDEX_KDTREE = 1
        self.flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
        self.hashfeatureTest = np.zeros(shape=[3, 1], dtype='float32')
        self.hashfeatureTest[1][0] = 1
        self.flann = cv2.flann.Index(self.hashfeatureTest, self.flann_params)
        np.save("D://devPy/hashfeature1",self.hashfeatureTest)
        print(np.load("D://devPy/hashfeature1.npy"))

        #kNN找最近邻,找候选集对应的id
        idx,_ = self.flann.knnSearch(feature1, K, params={})
        print(idx,_)
        #取出idx（去掉外层[]）中的所有元素(去内层[])
        candidate = idx[0].tolist()
        print(candidate)

        for i in range(K):
            # _condition=(feature1 == self.featureTest[candidate[i]])
            # condition = _condition.all(1)
            # res1=np.where(condition)
            # res2=res1[0]
            # print(_condition)
            # print(condition)
            # print(res1)
            # print(res2)
            same = np.where((feature1 == self.hashfeatureTest[candidate[i]]).all(1))[0]
            candidate.extend(same)
        candidate = list(set(candidate))
        print(candidate)
        return candidate

    def findByE(self,candidate,feature2):
        self.featureTest = np.zeros(shape=[3, 3], dtype='float32')
        self.featureTest[1][0] = 1
        self.featureTest[2][1] = 1
        #minDist=np.sqrt(np.sum(np.square(self.featureTest[candidate[0]] - feature2[0])))
        # label=candidate[0]
        # tag = candidate[0]
        # for i in range(1,len(candidate)):
        #     #feature2[0]为待检索图片的feature
        #     dist=np.sqrt(np.sum(np.square(self.featureTest[candidate[i]] - feature2[0])))
        #     if minDist>dist:
        #         minDist=dist
        #         label=tag
        #         tag=candidate[i]
        #如果检索的图片不在库中，无法返回不存在（因为没有保存dist）
        dist=lambda x: np.sum(np.square(self.featureTest[x] - feature2[0]))
        #dist = lambda x: np.linalg.norm(self.featureTest[x] - feature2[0])
        label=heapq.nsmallest(1, candidate, key=dist)
        print("lalalla",label)
        return label

    def find(self):
        img=self.readImg("C://Users/Mozhouting/Desktop/1.jpg")
        f1,f2=self.getFeatures(img)
        label=self.search(f1,f2)
        return

aa=Retrieval()
aa.find()

# a=[4,3,2,1]
# print(heapq.nsmallest(2,a,key=lambda x:-x))
