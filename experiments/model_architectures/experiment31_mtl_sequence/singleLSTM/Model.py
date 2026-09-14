import math
from torch import nn

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.init_weights()
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1)
        self.DNN = nn.Linear(self.hidden_size, 1)
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, log_return, time_mask):
        layer_out, layer_hidden = self.lstm(log_return)
        # layer_out.shape = [batch=32, seq=23, hidden=50]
        return self.DNN(layer_out[:,-1,:])

    def calculate_loss(self, criterion, outputs, labels):
        loss = criterion(outputs, labels)
        return loss