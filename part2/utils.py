import pprint as pp
import numpy as np
import math


def getData(fileName: str, height=25, width=10, interval=3,smooth=True):
    """
    get data from files
        :param fileName: path to file
        :param height=25: the height of matrix
        :param width=10: the width of matrix
        :param interval=3: the number of seperating line

    @return: the list of numpy matrix
    """
    file = open(fileName)
    ret = []

    def transform(line):
        line = line[:-1]
        ret = []
        for i in line:
            if i == '%':
                ret.append(0)
            else:
                ret.append(1)
        return ret

    data = list(map(transform, file.readlines()))

    num_datas = len(data) // (height + interval)

    for i in range(num_datas):
        ret.append(data[i * (height + interval):(i + 1) * (height + interval)])

    ret = list(map(lambda x: np.mat(x[:height], dtype=float), ret))
    if smooth:
        ret.append(np.ones((height, width)))
    return ret


def getPostProb(cond_prob: dict, data: list):
    '''
    @param: cond_prob: a dict in this form: {type: possibility matrix ...}
            data: a numpy matrix
    @return: the list of dict in this form: [{type: possibility...}, {...} ...]
    '''
    possibilities = []
    for kls, cond in cond_prob.items():
        possibility = 0
        for i in range(30):
            for j in range(13):
                if data[i, j] == 1.0:
                    try:
                        possibility += math.log2(cond[i, j])
                    except ValueError:
                        possibility += float('-inf')
                else:
                    try:
                        possibility += math.log2(1 - cond[i, j])
                    except ValueError:
                        possibility += float('-inf')
        possibilities.append((possibility, kls))
    return possibilities


def getCond_prob(datas: dict):
    """
    accept data and return cond_prob
        :param datas: a dict in this form: {type: a list of matrix....}
    """
    return {k: sum(v) / len(v) for k, v in datas.items()}


def getAccuracy(cond_prob, test):
    """
    get the confusionMatrix
        :param cond_prob: the conditional possibility dict
        :param test: test data and their tags, in this form: [(matrix, tag)]
    """
    confuseMatrix = {k1: {k2: 0 for k2 in cond_prob} for k1 in cond_prob}
    for i in test:
        # i[0] is data; i[1] is label
        possibilities = getPostProb(cond_prob, i[0])
        # add one to the confusion dict
        print(possibilities)
        my_guess = sorted(possibilities)[-1][1]
        print("MY GUESS: ", my_guess)
        print("TRUTH: ", i[1])
        confuseMatrix[i[1]][my_guess] += 1
    size = len(test)
    print(size)
    for k1 in confuseMatrix:
        for k2 in confuseMatrix[k1]:
            confuseMatrix[k1][k2]/=size
    return confuseMatrix

def getLabel(fileName):
    with open(fileName) as f:
        return [line.strip() for line in f.readlines()]