from splitData import split_all
import numpy as np
import torch
from torch.autograd import Variable

def tensorize_one(train_example):
    """
    :return: (10 * 25) tensor, label
    """
    mat, label = train_example
    mat = np.transpose(mat)
    tensor = Variable(torch.from_numpy(mat)) # 10 * 25 tensor
    className = 1 if label == 'yes' else 0
    t_class = torch.zeros(1)
    t_class[0] = className
    t_class = torch.unsqueeze(t_class, 0)
    return tensor, Variable(t_class)

def tensorize_all():
    data = split_all()
    for train_example in data:
        yield tensorize_one(train_example)

if __name__ == '__main__':
    data = tensorize_all()
    print(list(data))