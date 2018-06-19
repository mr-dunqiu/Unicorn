# -- coding:utf-8 --

def writeLines(filename:str, content):
    with open(filename, 'w') as file:
        file.writelines((str(line) + '\n') for line in content)

if __name__ == '__main__':
	pass