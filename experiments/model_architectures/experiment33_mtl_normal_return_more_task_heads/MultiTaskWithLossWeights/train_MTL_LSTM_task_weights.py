import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import Model

console = Console(width=90)
def train(sample_size_stocks:int, maximal_batch_size:int, hidden_size:int, initial_learning_rate:float, epochs:int):
    dataset = TechnicalIndicatorsDataSetFeaturesSplit(sample_size_stocks, "stationary_volume_new")
    path = os.path.join(f"outputs", "results.csv")

    for i in range(18):
        result_df = utils.create_results_csv([
            "hidden_size", "initial_learning_rate", "duration", "final_train_loss", "final_test_loss",
            "train_data_std", "test_data_std", "train_data_pos_movement_target", "test_data_pos_movement_target", "correct_direction"
        ], path)
        model = Model(hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-2)

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        callback = TrainCallback()
        final_train_loss, final_test_loss, final_percentage_devation_new, duration, model = utils.train_model_mtl(
            model, epochs, optimizer, train_loader, test_loader, scheduler, callbacks=[callback], train_step_fn=custom_train_step, test_step_fn=custom_test_step
        )
        callback.visualize_training_info()

        # compute and plot further statistics and distributions of the data
        test_mean, test_std, test_movement_direction, test_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(test_loader, labels_are_gross_return=True)
        train_mean, train_std, train_movement_direction, train_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(train_loader, labels_are_gross_return=True)
        summary_lines = [
            f"std_%: {np.round(final_percentage_devation_new, 3)}",
            f"[TRAIN] mean: {np.round(train_mean, 6)} | std: {np.round(train_std, 6)} | pos_movement: {np.round(train_movement_direction, 3)} | pos_movement_t: {np.round(train_movement_direction_target, 3)}",
            f"[TEST]  mean: {np.round(test_mean, 6)} | std: {np.round(test_std, 6)} | pos_movement: {np.round(test_movement_direction, 3)} | pos_movement_t: {np.round(test_movement_direction_target, 3)}"
        ]
        summary_text = "\n".join(summary_lines)
        console.print(Panel(summary_text, expand=False))
        console.print(Rule("\n TRAIN DATA DISTRIBUTIONS", style="magenta"))
        utils.print_time_and_days_distributions(train_loader)
        console.print(Rule("\n TEST DATA DISTRIBUTIONS", style="magenta"))
        utils.print_time_and_days_distributions(test_loader)
        model.print_task_weights()

        correct_directions = calculate_correct_direction_percentage(test_loader, model)
        print(f"Correct direction in %: {correct_directions}")

        # add the result to the existing ones
        result_df.loc[len(result_df)] = [
            hidden_size, initial_learning_rate, duration, final_train_loss, final_percentage_devation_new,
            train_std, test_std, train_movement_direction_target, test_movement_direction_target, correct_directions
        ]

        result_df.to_csv(os.path.join("outputs", "results.csv"), index=False)


def calculate_correct_direction_percentage(dataloader, model):
    model.eval()
    count, total_count = 0, 0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            total_count += labels.shape[0]
            outputs = model(**inputs)[:,3]
            labels = labels[:,3]
            #TODO muss das hier nicht angepasst werden bzgl. normalisierung und etc.?
            outputs_filtered = torch.where(outputs > 1, 1, 0)
            labels_filtered = torch.where(labels > 1, 1, 0)
            mask = outputs_filtered == labels_filtered
            result = outputs_filtered[mask]
            count += len(result.tolist())

    return count / total_count


#---------------------------------CUSTOM-TRAIN-FUNCTIONS-AND-CALLBACKS-------------------------------------------------#
class TrainCallback:
    def __init__(self):
        # store the train and test data in lists for later visualization
        self.train_batch_metrics, self.test_batch_metrics = [], []

    def on_train_batch_end(self, step_results, epoch):
        task_weights_grads = step_results.get("task_weights_gradients")
        row = {
            "epoch": epoch,
            "training_loss": step_results.get("running_train_loss"),
            # NOTE: grads aren't real averages of epoch, they are rather averages over the batches in an epoch
            "grad_norm_weights": step_results.get("gradient_norm"),
            "grad_stock_return": abs(task_weights_grads[0]),
            "grad_log_volume_change": abs(task_weights_grads[1]),
            "grad_rsi": abs(task_weights_grads[2]),
            "grad_stock_direction": abs(task_weights_grads[3]),
        }
        self.train_batch_metrics.append(row)


    def on_test_batch_end(self, step_results, epoch):
        diffs_array = step_results.get("diffs")
        task_losses = step_results.get("task_losses")
        row = {
            "epoch": epoch,
            "deviation_sum": np.sum(diffs_array),
            "batch_size": len(diffs_array),
            "stock_return_loss": task_losses[0],
            "log_volume_change_loss": task_losses[1],
            "rsi_loss": task_losses[2],
            "stock_direction_loss": task_losses[3],
            "total_loss": step_results.get("running_test_loss")
        }
        self.test_batch_metrics.append(row)


    def visualize_training_info(self):
        # Transforming the data
        df_train = pd.DataFrame(self.train_batch_metrics)
        df_test = pd.DataFrame(self.test_batch_metrics)

        df_train_epoch = df_train.groupby('epoch').mean().reset_index()
        df_test_epoch = df_test.groupby('epoch').sum().reset_index()
        # because the percentage deviation is desired -> multiply with 100 at the end
        df_test_epoch['percentage_deviation'] = (df_test_epoch['deviation_sum'] / df_test_epoch['batch_size']) * 100
        # everything above 1.5% (mostly at the early epochs) gets clipped -> better visualization
        df_test_epoch['percentage_deviation'] = df_test_epoch['percentage_deviation'].clip(upper=1.5)

        df_test_epoch['stock_loss_avg'] = (df_test_epoch['stock_return_loss'] / df_test_epoch['batch_size'])
        df_test_epoch['volume_change_loss_avg'] = (df_test_epoch['log_volume_change_loss'] / df_test_epoch['batch_size'])
        df_test_epoch['rsi_loss_avg'] = (df_test_epoch['rsi_loss'] / df_test_epoch['batch_size'])
        df_test_epoch['direction_loss_avg'] = (df_test_epoch['stock_direction_loss'] / df_test_epoch['batch_size'])
        df_test_epoch['total_loss_avg'] = (df_test_epoch['total_loss'] / df_test_epoch['batch_size'])

        # Visualization with Matplotlib
        self.plot_gradients(df_train_epoch, df_test_epoch)
        # due to different scales of task losses they are independently visualized
        self.plot_losses(df_test_epoch, "stock_loss_avg")
        self.plot_losses(df_test_epoch, "volume_change_loss_avg")
        self.plot_losses(df_test_epoch, "rsi_loss_avg")
        self.plot_losses(df_test_epoch, "direction_loss_avg")


    def plot_gradients(self, train_df, test_df):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('epochs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('percentage deviation (%)', color="#d62728", fontsize=12, fontweight='bold')

        line1 = ax1.plot(test_df['epoch'], test_df['percentage_deviation'], color="#d62728", marker='o', linewidth=2, label='Test Deviation (%)')
        ax1.tick_params(axis='y', labelcolor="#d62728")

        ax2 = ax1.twinx()
        ax2.set_ylabel('gradients (general norm & task-weights)', color="#1f77b4", fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor="#1f77b4")

        line2 = ax2.plot(train_df['epoch'], train_df['grad_norm_weights'],
                         color='black', linestyle='--', linewidth=2, label='Grad Norm (All Weights)')
        line3 = ax2.plot(train_df['epoch'], train_df['grad_stock_return'],
                         color='#1f77b4', label='Grad: Stock Return')
        line4 = ax2.plot(train_df['epoch'], train_df['grad_log_volume_change'],
                         color='#2ca02c', label='Grad: Volume Change')
        line5 = ax2.plot(train_df['epoch'], train_df['grad_rsi'],
                         color='#ff7f0e', label='Grad: RSI')
        line6 = ax2.plot(train_df['epoch'], train_df['grad_stock_direction'],
                         color='#9467bd', label='Grad: Stock Direction')

        ax2.tick_params(axis='y', labelcolor="#d62728")
        lines = line1 + line2 + line3 + line4 + line5 + line6
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=10)

        ax1.grid(True, linestyle=':', alpha=0.6)
        plt.title('training analysis: deviation vs. grads', fontsize=14, fontweight='bold')

        plt.tight_layout()  # tight-layout makes the legend better visible
        plt.show()


    def plot_losses(self, test_df, task_loss_name):
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('epochs', fontsize=12, fontweight='bold')
        ax1.set_ylabel('total loss', color="#d62728", fontsize=12, fontweight='bold')
        line1 = ax1.plot(test_df['epoch'], test_df['total_loss_avg'], color="#d62728", marker='o', linewidth=2, label='total loss')
        ax1.tick_params(axis='y', labelcolor="#d62728")

        ax2 = ax1.twinx()
        ax2.set_ylabel('task-specific losses', color="#1f77b4", fontsize=12, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor="#1f77b4")
        line2 = ax2.plot(test_df['epoch'], test_df[task_loss_name], color='black', linestyle='--', linewidth=2, label=task_loss_name)
        ax2.tick_params(axis='y', labelcolor="#d62728")
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True, fontsize=10)

        ax1.grid(True, linestyle=':', alpha=0.6)
        plt.title(f'training analysis: total vs. {task_loss_name}', fontsize=14, fontweight='bold')

        plt.tight_layout()  # tight-layout makes the legend better visible
        plt.show()


def custom_train_step(model_inputs:dict, labels, optimizer: optim.Optimizer, model, running_training_loss:float):
    optimizer.zero_grad()
    outputs = model.forward(**model_inputs)
    loss, _ = model.calculate_loss(outputs, labels)

    loss.backward()
    gradient_norm, task_weights_gradients = check_gradients_mtl_task_weightings(model)

    optimizer.step()
    running_training_loss += loss.item()

    train_output = {
        "running_train_loss": running_training_loss,
        "gradient_norm": gradient_norm,
        "task_weights_gradients": task_weights_gradients,
    }
    return train_output


def custom_test_step(model_inputs:dict, labels, model, running_testing_loss:float):
    outputs = model.forward(**model_inputs) # outputs.shape = [batch=32, outputs=4]
    loss, task_losses = model.calculate_loss(outputs, labels)
    task_losses_summed_up = torch.sum(task_losses, dim=0).detach().cpu().tolist()

    running_testing_loss += loss.item()
    diffs = torch.abs(outputs[:,0].squeeze(-1) - labels[:,0].squeeze(-1)).detach().cpu().numpy() # 1. output of batch = return
    test_output = {
        "running_test_loss": running_testing_loss,
        "diffs": diffs,
        "task_losses": task_losses_summed_up
    }
    return test_output


def check_gradients_mtl_task_weightings(model):
    total_norm = 0.0
    task_weights_gradients = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            if name == 'task_weights':
                task_weights_gradients = p.grad.detach().cpu().tolist()
            else:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm, task_weights_gradients


train (
    sample_size_stocks=23,
    maximal_batch_size=32,
    hidden_size=50,
    initial_learning_rate=0.001,
    epochs=100,
)