'''
'''
# coding = 'utf-8'
import sys
sys.path.append('d:/dev/unicorn/')
from src import Application
import FilesMaker
import DbBuild
from pathlib import PurePath


def fakeIlistFileEx(num,name,_):
    for i in range(num):
        yield name


class PerformanceEvaluater(object):
    def __init__(self,config):
        self.__actual = None
        self.__expected = None
        self.__config = config


    def evaluate(self):
        self.__getActual()
        self.__loadExpected()
        isOK, reason = self.__compare()
        print(reason if not isOK else 'ok')


    def __getActual(self):
        imageFinder = Application.ImageFinder(self.__config)
        print('Retrieval Start！')
        self.__actual ={}
        with open(self.__config['imagePathAlias'], 'r') as file:
            #WARNING!!!imageFinder.find()目前只能返回1个结果
            count = 0
            for imagePath, alias in (line.strip().split(' ') for line in file):
                self.__actual[alias] = imageFinder.find(imagePath)[0]
                count += 1
                import time
                time.sleep(0.0001)
                if count%100 == 0:
                    print(f'Processing Up To {count} Images', end='\r')
            print(f'Processed {count} Images Totally')

            #self.__actual = {alias:imageFinder.find(imagePath)[0] for imagePath,alias in (line.strip().split(' ') for line in file)}


    def __loadExpected(self):
        with open(self.__config['groundTruth'], 'r') as file:
            self.__expected = dict(line.strip().split(' ') for line in file)


    def __compare(self):
        print('Compute Accuracy Start!')
        if set(self.__actual.keys()) != set(self.__expected.keys()):
            return False,'keyNum is wrong!'
        # wrong = 0
        # for (k,v) in self.__actual.items():
        #     wrong += 1 if self.__expected[k] != v else 0
        # print(1-wrong/len(self.__expected))
        print('Compute Accuracy Completed ,is',len(set(self.__actual.items()) & set(self.__expected.items())) / len(self.__expected))

        return True,'OK'

from time import clock
if __name__ == '__main__':
    config = {
        #15张图
        #'imageRoot': r'D:\dev\unicorn\test\data\__NImage\image',
        #10k张图
        #'imageRoot': r'D:\1000000\image\images0\images\0',
        #100k张图
        'imageRoot': r'D:\1000000\image\images0\images',

        'targets': [(r'D:\dev\unicorn\test\data\__NImage\feature\imagePathLabels.txt', ['path', 'label']),
                    (r'D:\dev\unicorn\test\data\__NImage\feature\imagePathAlias.txt', ['path', 'alias']),
                    (r'D:\dev\unicorn\test\data\__NImage\feature\aliasLabel.txt', ['alias', 'label'])],
        'db': {
            'hashFeatureFilename': r'D:\dev\unicorn\test\data\__NImage\feature\1.bin',
            'imageFeatureFilename': r'D:\dev\unicorn\test\data\__NImage\feature\2.bin',
            'labelsFilename': r'D:\dev\unicorn\test\data\__NImage\feature\labels.txt',
        },

        'imagePathLabelsFilename': r'D:\dev\unicorn\test\data\__NImage\feature\imagePathLabels.txt',
        'groundTruth': r'D:\dev\unicorn\test\data\__NImage\feature\aliasLabel.txt',
        'imagePathAlias': r'D:\dev\unicorn\test\data\__NImage\feature\imagePathAlias.txt',
        # 'sample': [0, 10, 20]
        'extracter': {
            'num_classes': 11,
            'checkpoint_path': 'D:/dev/unicorn/model/model.ckpt-42100'
        }

    }

    start = clock()
    # step1
    def _resolve2(path):
        _path = PurePath(path)
        label = '_'.join([_path.parts[-2], _path.stem])
        imgAlias = '_'.join([label, _path.stem])
        return FilesMaker.ResolveResult(label, path, imgAlias)
    FilesMaker.makeConfigFile(config, resolver=_resolve2, openFileAfterWrite=True)
    finish=clock()
    print (finish-start)

    # WARNING!:imageFinder对象被重复构建
    # step2
    builder = DbBuild.DbBuilder(config)
    builder.build()
    finish=clock()
    print (finish-start)

    # step3
    #
    # PE = PerformanceEvaluater(config)
    # PE.evaluate()
    # finish=clock()
    # print (f'Time Consumed {finish-start} s')