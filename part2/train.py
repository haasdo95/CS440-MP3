import torch
from torch import nn
from torch import optim

from rnn import RNN
rnn = RNN(hidden_size=32)
rnn.train(mode=True)

from tensorize import tensorize_all
training_data = tensorize_all()
training_data = list(training_data)

criterion = nn.BCELoss()
optimizer = optim.SGD(rnn.parameters(), lr=0.001)

epoch = 30

for it in range(epoch):
    print("nth iter: ", it)
    for idx, (data, label) in enumerate(training_data):
        rnn.zero_grad()
        hidden = rnn.init_hidden()
        output = None
        for input in data:
            input = torch.unsqueeze(input, dim=0).float()
            output, hidden = rnn.forward(input, hidden)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        if (idx + 1) % 10 == 0:
            print("CURR: ", idx)
            print("GUESS: ", output)
            print("TRUTH: ", label)

torch.save(rnn.state_dict(), open('rnn.pkl', 'wb'))
