import math
import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from torch import nn

console = Console(width=90)

class Model(nn.Module):
    def __init__(self, hidden_size):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size, num_layers=1)
        self.shared_head = nn.Linear(self.hidden_size, 4)
        self.init_weights()
        self.s = nn.Parameter(torch.zeros(4))


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, log_return, time_mask, rsi, log_volume_change):
        feature = torch.cat((log_return, rsi, log_volume_change), dim=-1)  # [batch=32, seq=23, input=3]
        layer_out, layer_hidden = self.lstm(feature)  # layer_out.shape = [batch=32, seq=23, hidden=50]
        out = self.shared_head(layer_out[:, -1, :])
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

        weights = 0.5 * torch.exp(-self.s)
        weights[-1] = weights[-1] * 2 # the classification loss shouldn't be divided by 2
        weights_reshaped = weights[None, :] # shape=[1,4]
        weighted_tasks = (task_losses * weights_reshaped).sum(dim=1)
        final_loss = (weighted_tasks + (0.5*self.s.sum())).mean()
        return final_loss


    def _clip_s_grad(self, grad):
        """
            This function is registered in a hook and thus will be executed automatically
            Grad is the calculated gradient of the task weights
        """
        max_norm = 5.0
        norm = grad.norm(2)

        if norm > max_norm: grad = grad * (max_norm / (norm + 1e-4))

        return grad


    def print_task_weights(self):
        """
            Prints the learned task weights of the model.
            The smaller the task weights are the better and more important are the tasks
        """
        # weights = 0.5 * torch.exp(-self.s)
        weights_list = self.s.tolist()
        summary_lines = []
        for i, weight in enumerate(weights_list):
            summary_lines.append(f"Task {i+1} | Weight: {weights_list[i]}")

        summary_text = "\n".join(summary_lines)
        console.print(Rule("\n TASK WEIGHTS", style="magenta"))
        console.print(Panel(summary_text, expand=False))