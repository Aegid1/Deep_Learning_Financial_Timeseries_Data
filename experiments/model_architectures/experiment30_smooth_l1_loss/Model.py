from experiments.model_architectures.experiment30_smooth_l1_loss.custom_layers import MultiInputLSTMLessLSTMs as MILSTM_Less_LSTMs
import math
import torch
from torch import nn

class ModelLessLSTMs(nn.Module):
    def __init__(self, hidden_size):
        super(ModelLessLSTMs, self).__init__()
        self.hidden_size = hidden_size
        self.layers_lstms = nn.ModuleList()
        self.sequence_size = 23

        self.ema_dnn_layer = nn.Linear(in_features=self.sequence_size, out_features=16)
        self.ema_layer_norm = nn.LayerNorm(16)
        self.norm_layers = nn.ModuleList()

        # mask contains 5 categories/timesteps [0,1,2,3,4]
        self.mask_embedding = nn.Embedding(num_embeddings=5, embedding_dim=3)

        # stock_price, volume, rsi, macd, vwap
        for i in range(5):
            self.layers_lstms.append(nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=1))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

        self.MI_LSTM_layer = MILSTM_Less_LSTMs(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.milstm_norm = nn.LayerNorm(hidden_size)
        self.DNN_Layer = nn.Linear(in_features=hidden_size, out_features=50)
        # 16 * 1 because we insert here the other feature not processed through MI_LSTM -> ema
        self.DNN_Layer2 = nn.Linear(in_features=50+16, out_features=1)

        # these variables are just for debugging purposes after training
        self.attention_weights_last = None
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, log_return, log_volume_change, rsi, macd, vwap, ema, time_mask):
        features_not_lstms = [ema]
        features_lstms = [log_return, log_volume_change, rsi, macd, vwap]
        embedded_mask = self.mask_embedding(time_mask.to(dtype=torch.int)).squeeze(2) # [batch=16, seq=23, embedding=3]

        list_tilde = list()
        for i in range(len(self.layers_lstms)):
            feature = torch.cat((features_lstms[i], embedded_mask), dim=-1) # [batch=16, seq=23, input=4]
            layer_out, layer_hidden = self.layers_lstms[i](feature)
            layer_out = torch.tanh(self.norm_layers[i](layer_out))
            list_tilde.append(layer_out)

        milstm_tilde, milst_hidden = self.MI_LSTM_layer(
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3],
            list_tilde[4]
        ) # milstm_tilde [50, seq=23, 200]

        # process the mi-attention-lstm output through DNN
        milstm_tilde = self.dropout(torch.tanh(milstm_tilde))
        out_dnn1 = torch.relu(self.DNN_Layer(milstm_tilde))

        # process the ema just through DNN
        out_dnn2 = self.ema_dnn_layer(features_not_lstms[0].squeeze(-1)) # ema tensor shape [bs, seq_size, 1]
        dnn_2 = torch.tanh(self.ema_layer_norm(out_dnn2))

        combined_output = torch.cat([out_dnn1, dnn_2], dim=1)
        combined_output = self.dropout2(combined_output)
        out_dnn2 = self.DNN_Layer2(combined_output)
        return out_dnn2


    def calculate_loss(self, criterion, outputs, labels):
        loss = criterion(outputs, labels)
        return loss