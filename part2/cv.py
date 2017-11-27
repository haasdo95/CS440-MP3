from utils import *
from operator import itemgetter
import random
random.seed(666)

def cv(smoother: int):
    yes_train_no_sm = getData('./audioData/yes_train.txt', smooth=False)
    no_train_no_sm = getData('./audioData/no_train.txt', smooth=False)
    yes_train_sm = getData('./audioData/yes_train.txt', smooth=True, smoother=smoother)
    no_train_sm = getData('./audioData/no_train.txt', smooth=True, smoother=smoother)
    # random.seed(666)
    random.shuffle(yes_train_no_sm)
    # random.seed(666)
    random.shuffle(no_train_no_sm)
    # random.seed(666)
    random.shuffle(yes_train_sm)
    # random.seed(666)
    random.shuffle(no_train_sm)
    # random.seed(666)
    yes_cv_cnt = len(yes_train_no_sm) // 5
    no_cv_cnt = len(no_train_no_sm) // 5
    
    yes_cv = yes_train_no_sm[:yes_cv_cnt]
    yes_train = yes_train_sm[yes_cv_cnt:]

    no_cv = no_train_no_sm[:no_cv_cnt]
    no_train = no_train_sm[no_cv_cnt:]

    cond_prob = getCond_prob({'yes': yes_train, 'no': no_train})
    testData = [(i,'yes') for i in yes_cv]+[(i,'no') for i in no_cv]
    confuseMat = getAccuracy(cond_prob,testData,25,10)
    accuracy = 0
    for i in confuseMat:
        accuracy+=confuseMat[i][i]
    print("TOTAL ACCURACY:",accuracy)
    print(confuseMat)
    return accuracy


if __name__ == '__main__':
    smoothers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    record = {}
    for smoother in smoothers:
        accuracy = cv(smoother)
        print("using: ", smoother)
        print("having accuracy: ", accuracy)
        record[smoother] = accuracy
        print()
    lst = list(record.items())
    best_sm, accu = max(lst, key=itemgetter(1))
    print("Best Smoother: ", best_sm)
    print("Accu Best: ", accu)
