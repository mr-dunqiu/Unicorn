# -- coding:utf-8 --

import os

# pip install pypiwin32
import pythoncom
from win32com.shell import shell

def ilistFiles(dir):
	'''获取目录下的所有文件列表(generator)

	参数
	dir : 目录名
	'''

	_, _, files = next(os.walk(dir))
	return (os.path.join(dir, file) for file in files)

def listFiles(dir):
	'''获取目录下的所有文件列表(list)

	参数
	dir : 目录名
	'''

	_, _, files = next(os.walk(dir))
	#print([os.path.join(dir, file) for file in files])
	return [os.path.join(dir, file) for file in files]

def _getShorcutRealPath(path):
	try:
		pythoncom.CoInitialize()
		shortcut = pythoncom.CoCreateInstance(
						shell.CLSID_ShellLink,
						None,
						pythoncom.CLSCTX_INPROC_SERVER,
						shell.IID_IShellLink)
		shortcut.QueryInterface(pythoncom.IID_IPersistFile).Load(path)
		realPath = shortcut.GetPath(shell.SLGP_SHORTPATH)[0]
		return realPath
	except Exception as err:
		return path

def ilistFileEx(dir, recursive=True):
	'''获取目录下所有文件列表(generator)

	参数
	dir: 目录名
	'''
	for root, dirs, files in os.walk(dir):
		for file in files:
			# 不是.lnk文件，返回文件名
			if os.path.splitext(file)[1].lower() != '.lnk':
				yield os.path.normpath(os.path.join(root, file))
				continue

			# 是.lnk文件，但不是快捷方式，返回文件名
			realName = _getShorcutRealPath(os.path.join(root, file))
			if not realName:
				yield os.path.normpath(os.path.join(root, file))
				continue

			# 快捷方式指向文件，返回真实文件名
			if not os.path.isdir(realName):
				yield os.path.normpath(realName)
				continue

			# 快捷方式指向目录，则遍历该目录
			yield from ilistFileEx(realName)
		
		for dir in dirs:
			yield from ilistFileEx(dir)


if __name__ == '__main__':
	pass
	# i1 = [i for i in ilistFileEx(r'D:\dev\unicorn\test\data\__NImage\image')]
	# print(i1)