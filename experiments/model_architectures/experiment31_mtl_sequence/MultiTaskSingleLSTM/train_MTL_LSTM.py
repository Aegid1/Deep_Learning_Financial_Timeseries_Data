import os
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import Model

console = Console(width=90)
def train(sample_size_stocks:int, maximal_batch_size:int, hidden_size:int, initial_learning_rate:float, epochs:int, experiment_number: int):
    dataset = TechnicalIndicatorsDataSetFeaturesSplit(sample_size_stocks, "stationary_volume_new")
    final_percentage_devation = np.inf

    for i in range(90):
        result_df = utils.create_results_csv([
            "hidden_size", "initial_learning_rate", "duration", "final_train_loss", "final_test_loss",
            "train_data_std", "test_data_std", "train_data_pos_movement_target", "test_data_pos_movement_target"
        ])
        model = Model(hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)
        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        final_train_loss, final_test_loss, final_percentage_devation_new, duration, model = utils.train_model(
            model, epochs, nn.SmoothL1Loss(beta=3.5e-2, reduction="mean"), optimizer, train_loader, test_loader, scheduler
        )

        if final_percentage_devation_new < final_percentage_devation:  # only save the model when the std was reduced compared to previous model
            torch.save(model.state_dict(), os.path.join("outputs", f"mtl_single_lstm_{final_percentage_devation_new}.pth"))
            final_percentage_devation = final_percentage_devation_new

        # compute and plot further statistics and distributions of the data
        test_mean, test_std, test_movement_direction, test_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(test_loader)
        train_mean, train_std, train_movement_direction, train_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(train_loader)
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

        # add the result to the existing ones
        result_df.loc[len(result_df)] = [
            hidden_size, initial_learning_rate, duration, final_train_loss, final_percentage_devation_new,
            train_std, test_std, train_movement_direction_target, test_movement_direction_target
        ]

        result_df.to_csv(os.path.join("outputs", "results.csv"), index=False)

train (
    sample_size_stocks=23,
    maximal_batch_size=32,
    hidden_size=50,
    initial_learning_rate=0.005,
    epochs=100,
    experiment_number=30
)