import torch
from torch.autograd import Variable

from rnn import RNN
rnn = RNN(hidden_size=32)
rnn.train(mode=False)
s_dict = torch.load(open('rnn.pkl', 'rb'))
rnn.load_state_dict(s_dict)

import os
import utils
yes_path = './txt_yesno/yes_test'
no_path = './txt_yesno/no_test'
yesFiles = os.scandir(yes_path)
noFiles = os.scandir(no_path)

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
for t in test:
    total_count += 1
    mat_tensor, label = tensorize_one(t)
    my_guess = predict(mat_tensor, label.data[0][0])
    if my_guess == label.data[0][0]:
        # print("correct on: ", label.data[0][0])
        correct_count += 1

print("CORRECT: ", correct_count, " TOTAL: ", total_count)
print("RATE: ", correct_count / total_count)
