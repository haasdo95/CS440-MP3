"""
CREDIT TO: http://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
"""

import torch
from torch import nn
from torch.autograd import Variable

input_size = 25
output_size = 1

class RNN(nn.Module):
    def __init__(self, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)  # i + h -> h
        self.input2out = nn.Sequential(
            nn.Linear(input_size + hidden_size, output_size),
            nn.Sigmoid()
        )  # i + h -> 1

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        new_hidden = self.input2hidden(combined)
        out = self.input2out(combined)
        return out, new_hidden

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

if __name__ == "__main__":
    print(RNN(32))
