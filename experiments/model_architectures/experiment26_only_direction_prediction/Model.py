from custom_layers import MultiInputLSTMLessLSTMs as MILSTM_Less_LSTMs
import torch
from torch import nn

class ModelLessLSTMs(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ModelLessLSTMs, self).__init__()
        self.hidden_size = hidden_size

        self.layers_lstms_milstm = nn.ModuleList()
        self.layers_lstms = nn.ModuleList()
        self.dnns = nn.ModuleList()

        self.sequence_size = 23
        self.ema_dnn_layer = nn.Linear(in_features=self.sequence_size, out_features=16)
        self.norm_layers = nn.ModuleList()

        # stock_price, volume, rsi, macd, vwap
        for i in range(5):
            self.layers_lstms_milstm.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

        for i in range(8):
            self.layers_lstms.append(nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1))
            self.norm_layers.append(nn.LayerNorm(hidden_size))
            self.dnns.append(nn.Linear(hidden_size, 16))

        self.MI_LSTM_layer = MILSTM_Less_LSTMs(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.milstm_norm = nn.LayerNorm(hidden_size)
        #self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=50)
        # hidden_size + len(self.layers_lstms) * 3 because we insert here the other feature not processed through MI_LSTM
        add_features = len(self.layers_lstms) + 1
        self.DNN_Layer2 = nn.Linear(in_features=hidden_size+add_features*16, out_features=int((hidden_size+(add_features*16))))
        self.DNN_Layer3 = nn.Linear(in_features=int((hidden_size+(add_features*16))), out_features=1)

        # stock return direction layers
        self.return_direction_dnn1 = nn.Linear(in_features=hidden_size+(add_features*16), out_features=hidden_size+(add_features*16))
        self.return_direction_dnn2 = nn.Linear(in_features=hidden_size+(add_features*16), out_features=hidden_size+(add_features*16))
        self.return_direction_dnn3 = nn.Linear(in_features=hidden_size+(add_features*16), out_features=1)

        self.init_weights()

    def init_weights(self):
        #stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, param in self.named_parameters():
            #weight.data.uniform_(-stdv, stdv)
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.zero_()
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param.data)

    #features_lstm =  as a list [stock_prices, log_volume_change, rsi, macd, vwap, eur_usd, oil_price] -> 5
    #features = [ema] -> 1
    def forward(self, stock_prices, log_volume_change, rsi, macd, vwap, ema, eur_usd, oil_price, stoch_k, stoch_d, will_r, adosc, cci, mom):
        features_not_lstms = [ema]
        features_lstms_milstm = [stock_prices, log_volume_change, rsi, macd, vwap]
        features_lstms = [eur_usd, oil_price, stoch_k, stoch_d, will_r, adosc, cci, mom]
        list_tilde_milstm, list_tilde = list(), list()

        for i in range(len(self.layers_lstms_milstm)):
            layer_out, layer_hidden = self.layers_lstms_milstm[i](features_lstms_milstm[i])
            #layer_out = self.norm_layers[i](layer_out)
            list_tilde_milstm.append(layer_out)

        milstm_tilde, milst_hidden = self.MI_LSTM_layer(
            list_tilde_milstm[0],
            list_tilde_milstm[1],
            list_tilde_milstm[2],
            list_tilde_milstm[3],
            list_tilde_milstm[4]
        ) # milstm_tilde [50, 25, 200]
        #milstm_out = self.milstm_norm(milstm_tilde)
        milstm_out = self.dropout(torch.relu(milstm_tilde))

        for i in range(len(self.layers_lstms)):
            layer_out, layer_hidden = self.layers_lstms[i](features_lstms[i])
            layer_out = self.norm_layers[i+5](layer_out)
            if layer_out.dim() == 2: layer_out = layer_out.unsqueeze(0)
            layer_out = self.dnns[i](torch.relu(layer_out[:,-1,:]))
            list_tilde.append(layer_out)

        dnn_2 = self.ema_dnn_layer(features_not_lstms[0].squeeze(-1))
        if dnn_2.dim() == 1: # batch_size is 1 and was removed in previous step
            dnn_2 = dnn_2.unsqueeze(0)
        #out_dnn1 = self.DNN_Layer(torch.tanh(milstm_tilde))

        combined_output = torch.cat([
            milstm_out,
            dnn_2,
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3],
            list_tilde[4],
            list_tilde[5],
            list_tilde[6],
            list_tilde[7],
        ], dim=1)

        out_dnn2 = self.DNN_Layer2(torch.relu(combined_output))
        out_dnn3 = self.DNN_Layer3(out_dnn2)
        # direction_out = self.return_direction_dnn1(torch.relu(out_dnn3))
        # direction_out = self.return_direction_dnn2(direction_out)
        # direction_out = self.return_direction_dnn3(direction_out)

        return out_dnn3


    def calculate_loss(self, criterion, outputs, labels):
        loss = criterion(outputs, labels)
        return loss