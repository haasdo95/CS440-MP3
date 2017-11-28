import torch
from torch.autograd import Variable

from rnn import RNN
rnn = RNN(hidden_size=32)
rnn.train(mode=False)
s_dict = torch.load(open('./rnn.pkl', 'rb'))
rnn.load_state_dict(s_dict)

import os
import utils
yes_path = './txt_yesno/yes_test'
no_path = './txt_yesno/no_test'
yesFiles = os.scandir(yes_path)
noFiles = os.scandir(no_path)

import warnings
from viz import *

test = []
for i in yesFiles:
    test.append((utils.getData(i.path,interval=0,smooth=False)[0], 'yes'))
for i in noFiles:
    test.append((utils.getData(i.path,interval=0,smooth=False)[0], 'no'))

print("LEN: ", len(test))

def predict(mat, label):
    hidden = rnn.init_hidden()
    output_data = None
    for input_data in mat:
        input_data = torch.unsqueeze(input_data, dim=0).float()
        output_data, hidden = rnn.forward(input_data, hidden)
    output_data = output_data.data
    output_data = output_data[0][0]
    # print("PROB: ", output_data)
    # print("LABEL: ", label)
    if output_data > 0.5:
        return 1
    else:
        return 0

from tensorize import tensorize_one
total_count = 0
correct_count = 0
conf_mat = [[0, 0], [0, 0]]
for t in test:
    total_count += 1
    mat_tensor, label = tensorize_one(t)
    my_guess = predict(mat_tensor, label.data[0][0])
    conf_mat[int(label.data[0][0])][int(my_guess)] += 1
    if my_guess == label.data[0][0]:
        # print("correct on: ", label.data[0][0])
        correct_count += 1


print("CORRECT: ", correct_count, " TOTAL: ", total_count)
print("RATE: ", correct_count / total_count)

# plot the confusion matrix
# conf_mat = [[1,0], [0,1]] # row = ground truth, row/col --> NO, YES
# know open issue when using savefig()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    title = 'Confusion Matrix for Hebrew Words (Y/N) Using RNN'
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    file_name = str(parent_dir) + '../report/img/Hebrew_Words_RNN_Conf_Mat.png'
    plot_confusion_matrix(cm=conf_mat, classes=["NO", "YES"],
                          fname=file_name, normalize=True, title=title)
    print("CONF MAT GENERATED: ", title)
