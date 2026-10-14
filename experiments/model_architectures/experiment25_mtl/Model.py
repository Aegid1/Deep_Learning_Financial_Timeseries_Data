from custom_layers import MultiInputLSTMLessLSTMs as MILSTM_Less_LSTMs
import math
import torch
from torch import nn
from utils.experiment_utils import debug_activations

class ModelLessLSTMs(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelLessLSTMs, self).__init__()
        self.hidden_size = hidden_size
        self.layers_lstms = nn.ModuleList()
        self.sequence_size = 23

        self.oil_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.oil_dnn = nn.Linear(hidden_size, 16)
        self.oil_norm = nn.LayerNorm(16)

        self.currency_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.currency_dnn = nn.Linear(hidden_size, 16)
        self.currency_norm = nn.LayerNorm(16)

        self.ema_dnn_layer = nn.Linear(in_features=self.sequence_size, out_features=16)
        self.norm_layers = nn.ModuleList()

        #TODO: investigate if model performs better with macro factors processed in MI-LSTM

        # stock_price, volume, rsi, macd, vwap
        for i in range(5):
            self.layers_lstms.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

        self.MI_LSTM_layer = MILSTM_Less_LSTMs(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.milstm_norm = nn.LayerNorm(hidden_size)
        #self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=50)
        # hidden_size + 16 * 3 because we insert here the other feature not processed through MI_LSTM -> ema, oil, currency_rate
        self.DNN_Layer2 = nn.Linear(in_features=hidden_size+3*16, out_features=int((hidden_size+(3*16))/2))
        self.DNN_Layer3 = nn.Linear(in_features=int((hidden_size+(3*16))/2), out_features=int((hidden_size+(3*16))/2))

        # stock return prediction layers
        self.return_prediction_dnn1 = nn.Linear(in_features=int((hidden_size+(3*16))/2), out_features=int((hidden_size+(3*16))/2))
        self.return_prediction_dnn2 = nn.Linear(in_features=int((hidden_size+(3*16))/2), out_features=int((hidden_size+(3*16))/4))
        self.return_prediction_dnn3 = nn.Linear(in_features=int((hidden_size+(3*16))/4), out_features=1)

        # stock return direction layers
        self.return_direction_dnn1 = nn.Linear(in_features=int((hidden_size+(3*16))/2), out_features=int((hidden_size+(3*16))/2))
        self.return_direction_dnn2 = nn.Linear(in_features=int((hidden_size+(3*16))/2), out_features=int((hidden_size+(3*16))/4))
        self.return_direction_dnn3 = nn.Linear(in_features=int((hidden_size+(3*16))/4), out_features=2)

        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    #features_lstm =  as a list [stock_prices, log_volume_change, rsi, macd, vwap, eur_usd, oil_price] -> 5
    #features = [ema] -> 1
    def forward(self, stock_prices, log_volume_change, rsi, macd, vwap, ema, eur_usd, oil_price):
        features_not_lstms = [ema]
        features_lstms = [stock_prices, log_volume_change, rsi, macd, vwap]
        list_tilde = list()
        for i in range(len(self.layers_lstms)):
            layer_out, layer_hidden = self.layers_lstms[i](features_lstms[i])
            layer_out = torch.tanh(layer_out)

            list_tilde.append(layer_out)

        milstm_tilde, milst_hidden = self.MI_LSTM_layer(
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3],
            list_tilde[4]
        ) # milstm_tilde [50, 25, 200]

        milstm_out = self.milstm_norm(milstm_tilde)
        milstm_out = self.dropout(torch.relu(milstm_out))

        oil_tilde, oil_hidden = self.oil_lstm(oil_price)
        if oil_tilde.dim() == 2: oil_tilde = oil_tilde.unsqueeze(0)
        oil_out = torch.relu(self.oil_dnn(torch.tanh(oil_tilde[:,-1,:])))

        currency_rate_tilde, currency_rate_hidden = self.currency_lstm(eur_usd)
        if currency_rate_tilde.dim() == 2: currency_rate_tilde = currency_rate_tilde.unsqueeze(0)
        currency_out = torch.relu(self.currency_dnn(torch.tanh(currency_rate_tilde[:,-1,:])))

        dnn_2 = self.ema_dnn_layer(features_not_lstms[0].squeeze(-1))
        if dnn_2.dim() == 1: # batch_size is 1 and was removed in previous step
            dnn_2 = dnn_2.unsqueeze(0)

        #out_dnn1 = self.DNN_Layer(torch.tanh(milstm_tilde))

        combined_output = torch.cat([
            milstm_out,
            dnn_2,
            oil_out,
            currency_out
        ], dim=1)

        out_dnn2 = self.DNN_Layer2(combined_output)
        out_dnn3 = self.DNN_Layer3(torch.relu(out_dnn2))

        # stock return prediction task
        stock_return_out = self.return_prediction_dnn1(out_dnn3)
        stock_return_out = torch.relu(self.return_prediction_dnn2(stock_return_out))
        stock_return_out = self.return_prediction_dnn3(stock_return_out)

        # stock return direction task
        stock_direction_out = self.return_direction_dnn1(out_dnn3)
        stock_direction_out = torch.relu(self.return_direction_dnn2(stock_direction_out))
        stock_direction_out = self.return_direction_dnn3(stock_direction_out)

        return stock_return_out, stock_direction_out


    def calculate_loss(self, criterion, outputs, labels):
        loss = criterion(outputs, labels)
        return loss