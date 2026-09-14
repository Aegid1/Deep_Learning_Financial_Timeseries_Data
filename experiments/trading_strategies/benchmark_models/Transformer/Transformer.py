import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, **src):
        # inputs is always a dict which is here transformed into single tensor containing tensors
        feature_tensors = [tensor for tensor in src.values()]
        stacked_features = torch.cat(feature_tensors, dim=-1)

        # src [N, T, F], [512, 60, 6]
        stacked_features = self.feature_layer(stacked_features)  # [512, 60, 8]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        stacked_features = stacked_features.transpose(1, 0)  # not batch first

        mask = None

        stacked_features = self.pos_encoder(stacked_features)
        output = self.transformer_encoder(stacked_features, mask)  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])  # [512, 1]

        return output.squeeze()


    def calculate_loss(self, criterion, outputs, labels):
        return criterion(outputs, labels)