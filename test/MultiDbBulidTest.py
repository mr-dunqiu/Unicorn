'''
'''
# coding = 'utf-8'
import numpy as np
import os
import random
from src.Util import Path
from src import DbBuild
from src import FeatureExtract

class AbnormalDbBuilder(object):

	def __init__(self, config, featureExtracter=None):
		pass

	def build(self):
		imageFeatures = np.zeros(shape=[1, 128], dtype='float32')
		imageFeatures.tofile("D:/3.bin")
		#pass

# class DbBulidTester(object):
# 	def __init__(self):
# 		self.__config = None
# 		self.__hashFeature = None
# 		self.__imageFeature = None
# 		self.__featureExtracter = None
#
# 	def __setup(self):
# 		self.__config = {
# 			'imageRoot': 'D:/dev/unicorn/test/__1SingleImage/image/',
# 			'hashFeatureFilename': r'D:\dev\unicorn\test\__1SingleImage\feature\1.bin',
# 			'imageFeatureFilename': r'D:\dev\unicorn\test\__1SingleImage\feature\2.bin',
# 			#'sample':[],
# 			'extracter': {
# 				'num_classes': 11,
# 				'checkpoint_path': 'D:/dev/unicorn/model/model.ckpt-42100'
# 			}
# 		}
# 		self.__featureExtracter = FeatureExtract.FeatureExtracter(self.__config['extracter'])
# 		self.__hashFeature, self.__imageFeature = self.__featureExtracter.extract(r'D:\dev\unicorn\test\__1SingleImage\image\10.jpg')
#
#
# 	def __run(self):
# 		builder = DbBuild.DbBuilder(self.__config, self.__featureExtracter)
# 		#builder = AbnormalDbBuilder(self.__config, self.__featureExtracter)
# 		builder.build()
# 		isOK, reason= self.__compare()
# 		print(reason if not isOK else 'OK')
#
#
# 	def __compare(self):
#
# 		isOK, reason = self.__compareFeature(self.__config['imageFeatureFilename'], self.__imageFeature)
# 		if not isOK:
# 			return False, 'image feature ... , {}'.format(reason)
#
# 		isOK, reason = self.__compareFeature(self.__config['hashFeatureFilename'], self.__hashFeature)
# 		if not isOK:
# 			return False, 'hash feature ... , {}'.format(reason)
#
# 		return True, ''
#
#
# 	def __compareFeature(self, filename, expectedFeature):
#
# 		try:
# 			actualFeature = np.fromfile(filename, dtype=np.float32)
# 		except Exception:
# 			return False, "read file error!"
#
# 		if (expectedFeature.size != actualFeature.size):
# 			return False, "size not match!"
#
# 		if np.linalg.norm(actualFeature - expectedFeature) >= (1e-6):
# 			return False, "feature is wrong!"
#
# 		return True, "没毛病"
#
# 	def __cleanUp(self):
# 		try:
# 			os.remove(self.__config['imageFeatureFilename'])
# 			os.remove(self.__config['hashFeatureFilename'])
# 		except Exception as err:
# 			print(err)
#
# 	def test(self):
# 		self.__setup()
# 		self.__run()
# 		self.__cleanUp()

class MultiDbBuildTester(object):

	_OK = (True, '')
	_ERROR_BAD_FILE = (False, 'read file error')
	_ERROR_INCORRECT_DBSIZE = (False, 'size not match')
	_ERROR_INCORRENT_FEATURE = (False, 'feature is wrong')

	def __init__(self):
		self.__config = None
		self.__featureExtracter = None
		self.__totalImages = -1
		self.__checkCandidates = None
		self.__hashFeatures = None
		self.__imageFeatures = None

	def __setup(self):
		self.__config = {
			'imageRoot': r'D:\dev\unicorn\test\data\__NImage\image',
			'hashFeatureFilename': r'D:\dev\unicorn\test\data\__NImage\feature\1.bin',
			'imageFeatureFilename': r'D:\dev\unicorn\test\data\__NImage\feature\2.bin',
			# 'sample': [0, 10, 20]
			'extracter': {
				'num_classes': 11,
				'checkpoint_path': 'D:/dev/unicorn/model/model.ckpt-42100'
			}
		}
		self.__featureExtracter = FeatureExtract.FeatureExtracter(self.__config['extracter'])
		self.__setupFeatures()


	def __setupFeatures(self):
		imagePaths = Path.listFiles(self.__config['imageRoot'])
		self.__totalImages = len(imagePaths)
		self.__checkCandidates = self.__config.get('sample', MultiDbBuildTester._createCheckCandidates(self.__totalImages))
		testImageNum = len(self.__checkCandidates)
		self.__hashFeatures = np.zeros(shape=[testImageNum, 128], dtype='float32')
		self.__imageFeatures = np.zeros(shape=[testImageNum, 2048], dtype='float32')
		for i, j in enumerate(self.__checkCandidates):
			self.__hashFeatures[i], self.__imageFeatures[i] = self.__featureExtracter.extract(imagePaths[j])

	@staticmethod
	def _createCheckCandidates(totalImages):
		seq = range(totalImages)
		n = (totalImages + 9) // 10
		return random.sample(seq, n)

	def __run(self):
		builder = DbBuild.DbBuilder(self.__config, self.__featureExtracter)
		#builder = AbnormalDbBuilder(self.__config, self.__featureExtracter)
		builder.build()
		isOK, reason = self.__compare()
		print((reason, self.__checkCandidates) if not isOK else 'OK')


	def __compare(self):
		isOK, reason = self.__compareFeatures(self.__config['imageFeatureFilename'], 
											  self.__imageFeatures, 
											  self.__totalImages, 
											  2048)
		if not isOK:
			return False, 'image feature ... , {}'.format(reason)

		isOK, reason = self.__compareFeatures(self.__config['hashFeatureFilename'], 
											  self.__hashFeatures, 
											  self.__totalImages, 
											  128)
		if not isOK:
			return False, 'hash feature ... , {}'.format(reason)

		return MultiDbBuildTester._OK

	def __compareFeatures(self, filename, expectedFeatures, totalFeatures, featureDim):
		try:
			actualFeatures = np.fromfile(filename, dtype=np.float32)
		except Exception:
			return MultiDbBuildTester._ERROR_BAD_FILE

		if totalFeatures * featureDim != actualFeatures.size:
			return MultiDbBuildTester._ERROR_INCORRECT_DBSIZE

		actualFeatures.shape = totalFeatures, featureDim

		isOK = all([self.__compareFeature(actualFeatures[j], expectedFeatures[i]) 
					for i, j in enumerate(self.__checkCandidates)])
		return MultiDbBuildTester._OK if isOK else MultiDbBuildTester._ERROR_INCORRENT_FEATURE


	def __compareFeature(self, feature0, feature1):
		return np.linalg.norm(feature0 - feature1) < 1e-6


	def __cleanUp(self):
		try:
			os.remove(self.__config['imageFeatureFilename'])
			os.remove(self.__config['hashFeatureFilename'])
		except Exception as err:
			print(err)


	def test(self):
		self.__setup()
		self.__run()
		self.__cleanUp()



if __name__ == '__main__':
	Dbtester = MultiDbBuildTester()
	Dbtester.test()
	# imageRoot = config['imageRoot']
