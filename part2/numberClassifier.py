from utils import *
import numpy as np
import math


if __name__ == '__main__':
    raw_train_data = getData('./numberClassifier/training_data.txt',height=30,width=13,smooth=False)
    raw_test_data = getData('./numberClassifier/testing_data.txt',height=30,width=13)
    raw_train_lebal = getLabel('./numberClassifier/training_labels.txt')
    raw_test_lebal = getLabel('./numberClassifier/testing_labels.txt')
    # print(test_data)
    # print(test_lebal)
    train_data = {str(i):[np.ones((30,13))] for i in range(1,6)}
    # print(len(raw_train_data),len(raw_train_lebal))
    for i in range(len(raw_train_data)):
        train_data[raw_train_lebal[i]].append(raw_train_data[i])
    cond_prob = getCond_prob(train_data)
    test_data = [i for i in zip(raw_test_data,raw_test_lebal)]
    # print(cond_prob)
    confuseMat = getAccuracy(cond_prob,test_data)
    accuracy = 0
    for i in confuseMat:
        accuracy+=confuseMat[i][i]
    print(confuseMat)
    print("TOTAL ACCURACY:",accuracy)