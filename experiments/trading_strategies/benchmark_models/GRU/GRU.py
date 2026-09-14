import torch
from torch import nn


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)

        self.d_feat = d_feat


    def forward(self, **x):
        # inputs is always a dict which is here transformed into single tensor containing tensors
        feature_tensors = [tensor for tensor in x.values()]
        stacked_features = torch.cat(feature_tensors, dim=-1)

        out, _ = self.rnn(stacked_features)
        return self.fc_out(out[:, -1, :]).squeeze()


    def calculate_loss(self, criterion, outputs, labels):
        return criterion(outputs, labels)