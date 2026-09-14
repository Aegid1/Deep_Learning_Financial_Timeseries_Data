from custom_layers import MultiInputLSTMLessLSTMs as MILSTM_Less_LSTMs
import math
import torch
from torch import nn

class ModelLessLSTMs(nn.Module):
    def __init__(self, input_size, hidden_size, seq_size):
        super(ModelLessLSTMs, self).__init__()
        self.hidden_size = hidden_size
        self.layers_lstms = nn.ModuleList()
        # ema
        self.dnn_layer_1 = nn.Linear(in_features=seq_size, out_features=25)
        self.dnn_layer_2 = nn.Linear(in_features=25, out_features=1)
        self.norm_layers = nn.ModuleList()

        # stock_price, rsi, macd, vwap
        for i in range(4):
            self.layers_lstms.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

        self.MI_LSTM_layer = MILSTM_Less_LSTMs(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.milstm_norm = nn.LayerNorm(hidden_size)
        self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=50)
        # +1 because we insert here the other feature that isn't processed through LSTM
        self.DNN_Layer2 = nn.Linear(in_features=50+1, out_features=1)
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    #features_lstm =  as a list [stock_price, rsi, macd, vwap] -> 4
    #features = [ema] -> 1
    def forward(self, features_lstm, features):
        list_tilde, list_hidden = list(), list()
        for i in range(len(self.layers_lstms)):
            layer_out, layer_hidden = self.layers_lstms[i](features_lstm[:, i, :, :])
            layer_out = torch.tanh(layer_out)
            list_tilde.append(layer_out)
            list_hidden.append(layer_hidden)

        milstm_tilde, milst_hidden = self.MI_LSTM_layer(
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3]
        ) #milstm_tilde [50, seq_size, 200]
        milstm_tilde = self.dropout(torch.relu(milstm_tilde))

        dnn_outputs = list()
        dnn_1 = self.dnn_layer_1(features[:, 0, :, 0])
        dnn_2 = self.dnn_layer_2(torch.relu(dnn_1))
        dnn_outputs.append(dnn_2)

        dnn_outputs_tensor = torch.cat(dnn_outputs, dim=1)
        out_dnn1 = self.DNN_Layer(torch.tanh(milstm_tilde))
        combined_output = torch.cat([torch.relu(out_dnn1), torch.relu(dnn_outputs_tensor)], dim=1)
        out_dnn2 = self.DNN_Layer2(combined_output)
        return out_dnn2

    def calculate_loss(self, criterion, outputs, labels):
        loss = criterion(outputs, labels)
        return loss