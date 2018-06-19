'''
'''
# coding = 'utf-8'
from Util import Path
from Util import File
from pathlib import PurePath
import itertools
import operator
import numpy as np
import subprocess
import collections
import functools

ResolveResult = collections.namedtuple('ResolveResult', ['label', 'path', 'alias'])


@functools.lru_cache()
def _resolve(path):
    _path = PurePath(path)
    label = _path.parts[-2]
    imgAlias = '_'.join([label, _path.stem])
    return ResolveResult(label, path, imgAlias)


@functools.lru_cache()
def _extractFields(tp, fields):
    return ' '.join([_extractField(field, tp) for field in fields])


@functools.singledispatch
def _extractField(field, tp):
    pass


@_extractField.register(str)
def _(field, tp):
    return operator.attrgetter(field)(tp)


@_extractField.register(int)
def _(field, tp):
    return operator.itemgetter(field)(tp)


_RESOLVER_BY_NAME = {
    'default': _resolve
}


def makeConfigFile(config, resolver=None, openFileAfterWrite=False):
    imageRoot = config['imageRoot']
    resolver = resolver or _RESOLVER_BY_NAME[config.get('resolver', 'default')]

    for filename, fields in config['targets']:
        _fields = fields if isinstance(fields, tuple) else tuple(fields)


        seq = (_extractFields(resolver(imagePath), _fields) for imagePath in Path.ilistFileEx(imageRoot))

        File.writeLines(filename,seq)
        if openFileAfterWrite:
            subprocess.Popen(f'notepad {filename}', shell=True)


if __name__ == '__main__':
    # config = {
    #     'imageRoot': r'D:\dev\unicorn\test\data\__NImage\image',
    #     'targets': [(r'D:\dev\unicorn\test\data\__NImage\feature\imagePathLabels.txt', ['path','label']),
    #                 (r'D:\dev\unicorn\test\data\__NImage\feature\imagePathAlias.txt', ['path', 'alias']),
    #                 (r'D:\dev\unicorn\test\data\__NImage\feature\aliasLabel.txt',['alias','label'])]
    # }
    # # FM = FileMaker(config)
    # # FM.run()
    # makeConfigFile(config, openFileAfterWrite=True)


    pass
