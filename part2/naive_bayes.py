from dataLoader import yes_data, no_data, yes_test, no_test
import numpy as np
import math


def getPostProb(cond_prob, data):
    possibilities = []
    for kls, cond in cond_prob.items():
        possibility = 0
        for i in range(25):
            for j in range(10):
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

if __name__=='__main__':
    yes_prob = sum(yes_data) / len(yes_data)
    no_prob = sum(no_data) / len(no_data)
    # two condition probability matrix of two classes
    cond_prob = {'yes': yes_prob, 'no': no_prob}
    # print(cond_prob)
    count = 0
    same = 0
    yes_accu = 0
    no_accu = 0
    for i in no_test:
        possibilities = getPostProb(cond_prob, i)
        count += 1
        if sorted(possibilities)[-1][1] == 'no':
            same += 1
    no_accu = same / count
    print('no accuracy:', no_accu)
    for i in yes_test:
        possibilities = getPostProb(cond_prob, i)
        count += 1
        if sorted(possibilities)[-1][1] == 'yes':
            same += 1
    yes_accu = same / count
    print('yes accuracy:', yes_accu)

    print('total accuracy:', (yes_accu + no_accu) / 2)
