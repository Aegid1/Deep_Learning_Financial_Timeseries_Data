import math
import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from torch import nn
from custom_layers import MultiInputLSTMLessLSTMs as MILSTM_Less_LSTMs

console = Console(width=90)

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.sequence_size = 23
        self.layers_lstms = nn.ModuleList()

        self.ema_dnn_layer = nn.Linear(in_features=self.sequence_size, out_features=16)
        self.ema_layer_norm = nn.LayerNorm(16)
        self.norm_layers = nn.ModuleList()

        # mask contains 5 categories/timesteps [0,1,2,3,4]
        self.mask_embedding = nn.Embedding(num_embeddings=5, embedding_dim=3)

        # stock_return, log_volume_volume, rsi, macd, vwap
        for i in range(5):
            # input_size = 3-dim-embedding + 1-dim input
            self.layers_lstms.append(nn.LSTM(input_size=4, hidden_size=hidden_size, num_layers=1))
            self.norm_layers.append(nn.LayerNorm(hidden_size))

        self.mi_milstm_layer = MILSTM_Less_LSTMs(hidden_size, hidden_size)
        self.mi_lstm_norm = nn.LayerNorm(hidden_size)
        self.mi_lstm_dnn = nn.Linear(in_features=hidden_size, out_features=50)

        # +16 because we insert here the other feature not processed through MI_LSTM -> ema
        neurons_final_combined_layer = 50+16
        self.DNN_Layer2 = nn.Linear(in_features=neurons_final_combined_layer, out_features=neurons_final_combined_layer)

        self.dropout = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        # task specific heads
        self.stock_return_head = nn.Linear(neurons_final_combined_layer, 1)
        self.rsi_head = nn.Linear(neurons_final_combined_layer, 1)
        self.log_volume_change_head = nn.Linear(neurons_final_combined_layer, 1)
        self.return_direction_head = nn.Linear(neurons_final_combined_layer, 1)

        self.init_weights()
        self.task_weightings = nn.Parameter(torch.zeros(4))
        self.attention_weights_last = None


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, stock_return, time_mask, rsi, log_volume_change, macd, vwap, ema):
        features_not_lstms = [ema.squeeze(-1)]
        features_lstms = [stock_return, log_volume_change, rsi, macd, vwap]
        embedded_mask = self.mask_embedding(time_mask.to(dtype=torch.int)).squeeze(2)

        # first process the features through individual lstms
        list_tilde = list()
        for i in range(len(self.layers_lstms)):
            feature = torch.cat((features_lstms[i], embedded_mask), dim=-1)# [batch=32, seq=23, input=4]
            layer_out, layer_hidden = self.layers_lstms[i](feature)
            layer_out = torch.tanh(self.norm_layers[i](layer_out))
            list_tilde.append(layer_out)

        milstm_tilde, milst_hidden = self.mi_milstm_layer(
            list_tilde[0],
            list_tilde[1],
            list_tilde[2],
            list_tilde[3],
            list_tilde[4]
        )

        # process the mi-lstm output through DNN
        milstm_tilde = self.dropout(torch.tanh(milstm_tilde))
        out_milstm_dnn = torch.relu(self.mi_lstm_dnn(milstm_tilde))

        # process the ema just through DNN
        out_ema_dnn = self.ema_dnn_layer(features_not_lstms[0])
        out_ema_dnn = torch.tanh(self.ema_layer_norm(out_ema_dnn))

        combined_output = torch.cat([out_milstm_dnn, out_ema_dnn], dim=1)
        combined_output = self.dropout2(combined_output)
        combined_output = torch.tanh(self.DNN_Layer2(combined_output))

        # task outputs shapes = [batch=32, hidden=1]
        stock_return_out = self.stock_return_head(combined_output)
        rsi_out = self.rsi_head(combined_output)
        log_volume_change_out = self.log_volume_change_head(combined_output)
        return_direction_out = self.return_direction_head(combined_output)

        out = torch.cat((stock_return_out, log_volume_change_out, rsi_out, return_direction_out), dim=-1)
        return out


    def calculate_loss(self, outputs, labels):
        """
            Loss is implemented from paper: "(2018) Multi-Task Learning Using Uncertainty to Weigh Losses"
            As the authors described, the loss is calculated with self.s = torch.log(sigma**2)
        """
        mse, bce, smoothl1 = nn.MSELoss(reduction="none"), nn.BCEWithLogitsLoss(reduction="none"), nn.SmoothL1Loss(beta=3.5e-2, reduction="none")
        log_return_loss = smoothl1(outputs[:, 0], labels[:, 0]) # shape = [batch=32]
        log_volume_change_loss = mse(outputs[:, 1], labels[:, 1]) # shape = [batch=32]
        rsi_loss = mse(outputs[:, 2], labels[:, 2])  # shape = [batch=32]
        return_direction_loss = bce(outputs[:, 3], labels[:, 3]) # shape = [batch=32]

        task_losses = torch.cat((
            log_return_loss[:,None],
            log_volume_change_loss[:,None],
            rsi_loss[:, None],
            return_direction_loss[:, None]
            ), dim=-1
        ) # shape = [batch=32, tasks=4]

        weights = 0.5 * torch.exp(-self.task_weightings)
        weights[-1] = weights[-1] * 2 # the classification loss shouldn't be divided by 2
        weights_reshaped = weights[None, :] # shape=[1,4]
        weighted_tasks = (task_losses * weights_reshaped).sum(dim=1)
        final_loss = (weighted_tasks + (0.5*self.task_weightings.sum())).mean()
        return final_loss


    def print_task_weights(self):
        """
            Prints the learned task weights of the model.
            The smaller the task weights are the better and more important are the tasks
        """
        weights = 0.5 * torch.exp(-self.task_weightings)
        weights_list = weights.tolist()

        summary_lines = []
        for i, weight in enumerate(weights_list):
            if i == (len(weights_list) - 1):
                summary_lines.append(f"Task {i + 1} | Weight: {weights_list[i] * 2}")
            else:
                summary_lines.append(f"Task {i + 1} | Weight: {weights_list[i]}")

        summary_text = "\n".join(summary_lines)
        console.print(Rule("\n TASK WEIGHTS", style="magenta"))
        console.print(Panel(summary_text, expand=False))