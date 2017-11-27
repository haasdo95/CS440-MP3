import utils
import numpy as np
import math, warnings, os
from visualization.viz import *

def avg_col(datas):
    '''
    take a list of numpy matrix and reduce each column of each matrix to one number
    '''
    ret = []
    for i in datas:
        row, col = i.shape
        l = []
        for r in range(row):
            sum = 0
            for c in range(col):
                sum += i[r, c]
            l.append(sum)
        ret.append(l)
    return ret


def collect(datas):
    ret = {tag: [{float(j): 1 for j in range(11)}
                 for i in range(25)] for tag in ['yes', 'no']}
    for k in datas:
        dataSetSize = len(datas[k])
        for i in datas[k]:
            for pos, j in enumerate(i):
                ret[k][pos][j] += 1
        for i in range(25):
            for j in range(11):
                ret[k][i][float(j)] /= dataSetSize
    return ret


def getPostProb(cond_prob, data):
    possibilities = []
    for kls, cond in cond_prob.items():
        possibility = 0
        for i in range(25):
            possibility += math.log2(cond[i][data[i]])
        possibilities.append((possibility, kls))
    return possibilities


def getAccuracy(cond_prob, test):
    """
    get the confusionMatrix
        :param cond_prob: the conditional possibility dict
        :param test: test data and their tags, in this form: [(matrix, tag)]
        :param height: height of input matrix
        :param width: width of input matrix
    """
    confuseMatrix = {k1: {k2: 0 for k2 in cond_prob} for k1 in cond_prob}
    for i in test:
        # i[0] is data; i[1] is label
        possibilities = getPostProb(cond_prob, i[0])
        # add one to the confusion dict
        my_guess = sorted(possibilities)[-1][1]
        # print("MY GUESS: ", my_guess)
        # print("TRUTH: ", i[1])
        confuseMatrix[i[1]][my_guess] += 1
    size = len(test)
    for k1 in confuseMatrix:
        for k2 in confuseMatrix[k1]:
            confuseMatrix[k1][k2] /= size
    return confuseMatrix


if __name__ == "__main__":
    yes_train = utils.getData('./audioData/yes_train.txt', smooth=False)
    no_train = utils.getData('./audioData/no_train.txt', smooth=False)
    print('TOTAL TRAIN SIZE:', len(yes_train) + len(no_train))
    no_test = utils.getData('./audioData/no_test.txt', smooth=False)
    yes_test = utils.getData('./audioData/yes_test.txt', smooth=False)
    yes_train = avg_col(yes_train)
    no_train = avg_col(no_train)
    train_data = {'yes': yes_train, 'no': no_train}
    condProb = collect(train_data)
    test_data = [(i, 'yes') for i in avg_col(yes_test)] + \
        [(i, 'no') for i in avg_col(no_test)]
    confuseMat = getAccuracy(condProb, test_data)
    print(confuseMat)
    conf_mat = list([[val for _, val in val_dict.items()] for _, val_dict in confuseMat.items()])
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    # know open issue when using savefig()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        title = 'Confusion Matrix for Hebrew Words (Y/N) Using Average Column Method'
        file_name = str(parent_dir) + '/report/img/Hebrew_Words_Avg_Col_Conf_Mat.png'
        plot_confusion_matrix(cm=conf_mat, classes=list([key.upper() for key in confuseMat.keys()]),
                              fname=file_name, normalize=True, title=title)
        print("CONF MAT GENERATED: ", title)
