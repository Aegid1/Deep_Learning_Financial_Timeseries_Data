import torch
from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, **kwargs):
        # input features get unpacked into params and these arguments get combined into single tensor
        x = torch.cat(list(kwargs.values()), dim=-1)

        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        prediction = self.fc(last_time_step)

        return prediction

    def calculate_loss(self, criterion, outputs, labels):
        return criterion(outputs, labels)
