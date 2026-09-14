from custom_layers import MultiInputLSTMLessLSTMs as MILSTM_Less_LSTMs
import math
import torch
from torch import nn
from utils.experiment_utils import debug_activations

class ModelLessLSTMs(nn.Module):
    def __init__(self, hidden_size):
        super(ModelLessLSTMs, self).__init__()
        self.hidden_size = hidden_size
        self.layers_lstms = nn.ModuleList()
        self.sequence_size = 23
        self.ema_dnn_layer = nn.Linear(in_features=self.sequence_size, out_features=16)
        self.norm_layers = nn.ModuleList()
        # mask contains 5 categories/timesteps [0,1,2,3,4]
        self.mask_embedding = nn.Embedding(num_embeddings=5, embedding_dim=3)
        # stock_price, volume, rsi, macd, vwap, oil_change, currency_change
        for i in range(7):
            self.layers_lstms.append(nn.LSTM(input_size=4, hidden_size=hidden_size, batch_first=True))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

        self.MI_LSTM_layer = MILSTM_Less_LSTMs(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.milstm_norm = nn.LayerNorm(hidden_size)
        self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=50)
        # 16 * 1 because we insert here the other feature not processed through MI_LSTM -> ema
        self.DNN_Layer2 = nn.Linear(in_features=50+16, out_features=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    #features_lstm =  as a list [log_return, log_volume_change, rsi, macd, vwap, oil_change, currency_change] -> 7
    #features = [ema] -> 1
    def forward(self, log_return, log_volume_change, rsi, macd, vwap, ema, oil_price, eur_usd, time_mask):
        features_not_lstms = [ema]
        features_lstms = [log_return, log_volume_change, rsi, macd, vwap, oil_price, eur_usd]
        list_tilde = list()
        embedded_mask = self.mask_embedding(time_mask.to(dtype=torch.int)).squeeze(2) # [batch=16, seq=23, embedding=3]
        for i in range(len(self.layers_lstms)):
            feature = torch.cat((features_lstms[i], embedded_mask), dim=-1) # [batch=16, seq=23, input=4]
            layer_out, layer_hidden = self.layers_lstms[i](feature)
            layer_out = self.norm_layers[i](layer_out)
            layer_out = torch.tanh(layer_out)
            list_tilde.append(layer_out)

        milstm_tilde, milst_hidden = self.MI_LSTM_layer(
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3],
            list_tilde[4],
            list_tilde[5],
            list_tilde[6],
        )

        milstm_tilde = self.dropout(torch.relu(milstm_tilde))

        dnn_2 = torch.relu(self.ema_dnn_layer(features_not_lstms[0].squeeze(-1)))
        if dnn_2.dim() == 1: # batch_size is 1 and was removed in previous step
            dnn_2 = dnn_2.unsqueeze(0)
        out_dnn1 = self.DNN_Layer(milstm_tilde)

        combined_output = torch.cat([
            out_dnn1,
            dnn_2
        ], dim=1)

        out_dnn2 = self.DNN_Layer2(combined_output)
        return out_dnn2


    def calculate_loss(self, criterion, outputs, labels):
        loss = criterion(outputs, labels)
        return loss