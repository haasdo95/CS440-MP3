import numpy as np
from utils import *
import os
path = './txt_yesno/training'
files = os.scandir(path)


def split(filePath, preLen, postLen, startThreshold, endThreshold):
    '''
    This function return a list [(data1, tag1), (data2, tag2), ...] 
    '''
    mat = getData(filePath, height=25, width=150, interval=0, smooth=False)[0]
    mat = np.transpose(mat)[preLen:-postLen]
    usefulData = []
    isWord = False
    word = np.ndarray((0, 25))
    _, fileName = os.path.split(filePath)
    types = ['yes' if x == '1' else 'no' for x in
             fileName.split('.txt')[0].split('_')]
    # print(types)
    for i in mat:
        highEnergy = 0
        for j in i.tolist()[0]:
            if j == 1.0:
                highEnergy += 1
            if highEnergy == startThreshold and not isWord:
                isWord = True
        if (isWord and highEnergy < endThreshold) or len(word) == 10:
            isWord = False
            if len(word) < 10:
                padLen = 10 - len(word)
                for i in range(padLen // 2):
                    word = np.concatenate((np.zeros((1, 25)), word))
                for i in range(padLen - padLen // 2):
                    word = np.concatenate((word, np.zeros((1, 25))))
                # word = np.pad(word,(padLen//2,padLen-padLen//2),'constant',constant_values=(np.zeros((1,25)), np.zeros((1,25))))

            usefulData.append(word.transpose())
            word = np.ndarray((0, 25))
        if isWord:
            # print(i)
            # print(word)
            word = np.concatenate((word, i), axis=0)
    if len(usefulData) > 8:
        return None
    #     output(usefulData, 'badExample' + fileName)
    # for i in types
    return [i for i in zip(usefulData,types)]


def output(datas, fileName):
    with open(fileName, 'w') as f:
        for i in datas:
            h, w = i[0].shape
            for x in range(h):
                for y in range(w):
                    if i[0][x, y] == 1.0:
                        f.write(' ')
                    else:
                        f.write('%')
                f.write('\n')
            f.write('\n\n\n')

def formalData(datas):
    """return the formal data"""
    d = {}
    for i,ty in datas:
        if ty not in d:
            d[ty] = [np.zeros(i.shape),np.ones(i.shape)]
        d[ty].append(i)
    return d

def run():
    datas = []
    for fileEntry in files:
        result = split(fileEntry.path, 10, 10, 4, 2)
        if result:
            datas += result
    return getCond_prob(formalData(datas))

if __name__ == "__main__":
    datas = []
    for fileEntry in files:
        result = split(fileEntry.path, 10, 10, 4, 2)
        if result:
            print(result)
            datas += result
    output(datas, 'output.txt')
    print(getCond_prob(formalData(datas)))

# usefulData = split(file1, 10, 10, 3, 1)
# for i in usefulData:
#     print(i.shape)
