import math
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=1)
        self.shared_head = nn.Linear(self.hidden_size, 4) # TODO maybe use individual heads
        self.init_weights()
        self.s = nn.Parameter(torch.zeros(4))

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, log_return, time_mask, rsi, log_volume_change):
        feature = torch.cat((log_return, rsi, log_volume_change), dim=-1)  # [batch=32, seq=23, input=3]
        layer_out, layer_hidden = self.lstm(feature) # layer_out.shape = [batch=32, seq=23, hidden=50]
        out = self.shared_head(layer_out[:,-1,:])
        return out

    def calculate_loss(self, outputs, labels):
        mse, bce, smoothl1 = nn.MSELoss(), nn.BCEWithLogitsLoss(), nn.SmoothL1Loss(beta=3.5e-2)
        final_loss = (
            smoothl1(outputs[:,0], labels[:,0]) + # log_return
            mse(outputs[:,1], labels[:,1]) + # log_volume_change
            mse(outputs[:,2], labels[:,2]) + # rsi
            bce(outputs[:,3], labels[:,3]) # return_direction
        ) / 4
        return final_loss
