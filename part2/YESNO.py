# from dataLoader import yes_data, no_data, yes_test, no_test
from utils import *
import numpy as np
import math, warnings, os
from visualization.viz import *


if __name__ == '__main__':
    yes_train = getData('./audioData/yes_train.txt', smoother=2)
    no_train = getData('./audioData/no_train.txt', smoother=2)
    print('TOTAL TRAIN SIZE:',len(yes_train)+len(no_train))
    no_test = getData('./audioData/no_test.txt',smooth=False)
    yes_test = getData('./audioData/yes_test.txt',smooth=False)
    cond_prob = getCond_prob({'yes': yes_train, 'no': no_train})
    testData = [(i,'yes') for i in yes_test]+[(i,'no') for i in no_test]
    confuseMat = getAccuracy(cond_prob,testData,25,10)
    accuracy = 0
    for i in confuseMat:
        accuracy+=confuseMat[i][i]
    print("TOTAL ACCURACY:",accuracy)
    print(confuseMat)
    conf_mat = list([[val for _, val in val_dict.items()] for _, val_dict in confuseMat.items()])
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    # know open issue when using savefig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        title = 'Confusion Matrix for Hebrew Words (Y/N)'
        file_name = str(parent_dir) + '/report/img/Hebrew_Words_Conf_Mat.png'
        plot_confusion_matrix(cm=conf_mat, classes=list([key.upper() for key in confuseMat.keys()]),
                              fname=file_name, normalize=True, title=title)
        print("CONF MAT GENERATED: ", title)