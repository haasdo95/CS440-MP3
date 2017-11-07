# from dataLoader import yes_data, no_data, yes_test, no_test
from utils import *
import numpy as np
import math


if __name__ == '__main__':
    yes_train = getData('./audioData/yes_train.txt')
    no_train = getData('./audioData/no_train.txt')
    print('TOTAL TRAIN SIZE:',len(yes_train)+len(no_train))
    no_test = getData('./audioData/no_test.txt',smooth=False)
    yes_test = getData('./audioData/yes_test.txt',smooth=False)
    cond_prob = getCond_prob({'yes': yes_train, 'no': no_train})
    testData = [(i,'yes') for i in yes_test]+[(i,'no') for i in no_test]
    confuseMat = getAccuracy(cond_prob,testData)
    accuracy = 0
    for i in confuseMat:
        accuracy+=confuseMat[i][i]
    print("TOTAL ACCURACY:",accuracy)
