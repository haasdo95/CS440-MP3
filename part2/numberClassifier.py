from utils import *
import numpy as np
import math, warnings, os
from visualization.viz import *

if __name__ == '__main__':
    raw_train_data = getData('./numberClassifier/training_data.txt',height=30,width=13,smooth=False)
    raw_test_data = getData('./numberClassifier/testing_data.txt',height=30,width=13,smooth=False)
    raw_train_lebal = getLabel('./numberClassifier/training_labels.txt')
    raw_test_lebal = getLabel('./numberClassifier/testing_labels.txt')
    train_data = {str(i):[np.ones((30,13)),np.zeros((30,13))] for i in range(1,6)}
    for i in range(len(raw_train_data)):
        train_data[raw_train_lebal[i]].append(raw_train_data[i])
    cond_prob = getCond_prob(train_data)
    test_data = [i for i in zip(raw_test_data,raw_test_lebal)]
    confuseMat = getAccuracy(cond_prob,test_data,30,13)
    accuracy = 0
    for i in confuseMat:
        accuracy+=confuseMat[i][i]
    print(confuseMat)
    print("TOTAL ACCURACY:",accuracy)
    for _, v in confuseMat.items():
        for k in v:
            v[k] *= 5
    print(confuseMat)
    conf_mat = [[confuseMat[str(i)][str(j)] for j in range(1, 6)] for i in range(1, 6)]
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    # know open issue when using savefig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        title = 'Confusion Matrix for Audio Digits'
        file_name = str(parent_dir) + '/report/img/Audio_Digits_Conf_Mat.png'
        plot_confusion_matrix(cm=conf_mat, classes=["1", "2", "3", "4", "4"],
                              fname=file_name, normalize=True, title=title)
        print("CONF MAT GENERATED: ", title)