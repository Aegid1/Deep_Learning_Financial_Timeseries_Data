import os
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import Model

console = Console(width=90)
def train(sample_size_stocks:int, maximal_batch_size:int, hidden_size:int, initial_learning_rate:float, epochs:int):
    dataset = TechnicalIndicatorsDataSetFeaturesSplit(sample_size_stocks, "stationary_volume_new")
    path = os.path.join(f"outputs", "results.csv")

    for i in range(3):
        result_df = utils.create_results_csv([
            "hidden_size", "initial_learning_rate", "duration", "final_train_loss", "final_test_loss",
            "train_data_std", "test_data_std", "train_data_pos_movement_target", "test_data_pos_movement_target", "correct_direction"
        ], path)
        model = Model(hidden_size)
        # optimizer = torch.optim.AdamW([
        #     {'params': model.lstm.parameters(), 'lr': 0.005},
        #     {'params': model.shared_head.parameters(), 'lr': 0.005},
        #     {'params': [model.s], 'lr': 0.0001}],
        #     weight_decay=1e-2
        # )
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-2)

        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        final_train_loss, final_test_loss, final_percentage_devation_new, duration, model = utils.train_model_mtl(
            model, epochs, optimizer, train_loader, test_loader, scheduler
        )

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

            outputs_filtered = torch.where(outputs > 0, 1, 0)
            labels_filtered = torch.where(labels > 0, 1, 0)
            mask = outputs_filtered == labels_filtered
            result = outputs_filtered[mask]
            count += len(result.tolist())

    return count / total_count


train (
    sample_size_stocks=23,
    maximal_batch_size=32,
    hidden_size=50,
    initial_learning_rate=0.0005,
    epochs=100,
)