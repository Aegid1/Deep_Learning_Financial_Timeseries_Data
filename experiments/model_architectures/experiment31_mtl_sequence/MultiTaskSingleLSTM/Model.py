import math

import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.init_weights()
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1)
        self.shared_head = nn.Linear(self.hidden_size, 1)
        # these are used for calculating the multitask loss
        self.log_return = None
        self.out = None

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, log_return, time_mask):
        layer_out, layer_hidden = self.lstm(log_return)  # layer_out.shape = [batch=32, seq=23, hidden=50]
        # applies the final linear layer to every output of each timestep
        out = self.shared_head(layer_out)

        self.log_return = log_return
        self.out = out # out.shape = [batch=32, tasks=23, output=1]
        return out[:,-1,:]

    def calculate_loss(self, criterion, outputs, labels):
        labels_mtl = torch.cat((self.log_return[:,1:,:], labels.unsqueeze(-1).unsqueeze(-1)), dim=1)
        loss = criterion(self.out, labels_mtl)
        return loss
