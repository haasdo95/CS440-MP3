import numpy as np
from utils import *
path = './txt_yesno/training'

file1 = path + '/0_0_0_0_1_1_1_1.txt'


def split(fileName, preLen, postLen, threshold):
    mat = getData(fileName, height=25, width=150, interval=0, smooth=False)[0]
    mat = np.transpose(mat)
    print(mat)
    print(mat.shape)
    usefulData = []
    isWord = False
    word = np.mat([])
    for i in mat:
        highEnergy = 0
        print(i)
        for j in i:
            if j == 1.0:
                highEnergy += 1
            if highEnergy == threshold and not isWord:
                isWord = True
            if isWord and highEnergy<threshold or len(word)==10:
                isWord = False
                usefulData.append(word)
                word = np.mat([])
        if isWord:
            word.concatenate(i)
    return usefulData

split(file1,10,10,3)