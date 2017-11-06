import pprint as pp
import numpy as np
yes_train = open('./audioData/yes_train.txt')
no_train = open('./audioData/no_train.txt')
no_test = open('./audioData/no_test.txt')
yes_test = open('./audioData/yes_test.txt')

def getData(file):
    ret = []
    def transform(line):
        line = line[:-1]
        ret = []
        for i in line:
            if i=='%':
                ret.append(0)
            else:
                ret.append(1)
        return ret

    data = list(map(transform,file.readlines()))

    num_datas = len(data) // 28

    for i in range(num_datas):
        ret.append(data[i * 28:(i + 1) * 28])


    ret = list(map(lambda x: np.mat(x[:25],dtype=float),ret))
    ret.append(np.ones((25,10)))
    return ret


yes_data = getData(yes_train)
no_data = getData(no_train)
yes_test = getData(yes_test)
no_test = getData(no_test)
if __name__ == "__main__":
    pp.pprint(yes_data)
    pp.pprint(no_data)
    pp.pprint(yes_test)
    pp.pprint(no_test)