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
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size, num_layers=1)
        self.shared_head = nn.Linear(self.hidden_size, 1)
        self.s = nn.Parameter(torch.zeros(23))
        self.init_weights()
        # these are used for calculating the multitask loss
        self.log_return = None
        self.multi_task_out = None


    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, log_return, time_mask):
        layer_out, layer_hidden = self.lstm(log_return)  # layer_out.shape = [batch=32, seq=23, hidden=50]
        # applies the final linear layer to every output of each timestep
        out = self.shared_head(layer_out)

        self.log_return = log_return
        self.multi_task_out = out # shape = [batch=32, tasks=23, output=1]
        return out[:,-1,:]


    def calculate_loss(self, criterion, outputs, labels):
        """
            Loss is implemented from paper: "(2018) Multi-Task Learning Using Uncertainty to Weigh Losses"
            As the authors described, the loss is calculated with self.s = torch.log(sigma**2)
        """

        labels_mtl = torch.cat((self.log_return[:,1:,:], labels[:, None, None]), dim=1) # shape=[32, 23, 1]
        task_losses = criterion(self.multi_task_out, labels_mtl) # shape=[32, 23, 1]

        weights = 0.5 * torch.exp(-self.s)
        weights_reshaped = weights[None, :, None] # shape=[1,23,1]
        weighted_tasks = (task_losses * weights_reshaped).sum(dim=1)
        final_loss = (weighted_tasks + (0.5*self.s.sum())).mean()
        return final_loss


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