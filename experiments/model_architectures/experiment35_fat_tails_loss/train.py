import os
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils
from exponential_weights.Dataset import TechnicalIndicatorsDataSetFeaturesSplit as ExponentialWeightsDataset
from exponential_weights.Model import ModelLessLSTMs as ExponentialWeightsModel
from tercile_weights.Model import ModelLessLSTMs as TercileWeightsModel
from tercile_weights.Dataset import TechnicalIndicatorsDataSetFeaturesSplit as TercileWeightsDataset


console = Console(width=90)
def train(
        maximal_batch_size:int,
        hidden_size:int,
        initial_learning_rate:float,
        epochs:int,
        experiment_number: int,
        dataset,
        model_factory,
        results_path
):
    results_df_path = os.path.join(results_path, "results.csv")
    for i in range(20):
        model = model_factory()
        result_df = utils.create_results_csv(
            columns=["hidden_size", "params", "initial_learning_rate", "duration", "final_train_loss", "final_test_loss",
            "train_data_std", "test_data_std", "train_data_pos_movement_target", "test_data_pos_movement_target"],
            path=results_df_path
        )

        params = sum(p.numel() for p in model.parameters() if p.requires_grad) # calculates total amount of params
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)
        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        final_train_loss, final_test_loss, final_percentage_devation, duration, model = utils.train_model(
            model, epochs, nn.MSELoss(), optimizer, train_loader, test_loader, scheduler
        )

        torch.save(model.state_dict(), os.path.join(results_path, f"trained_model_experiment{experiment_number}_{final_percentage_devation}.pth"))

        # compute and plot further statistics and distributions of the data
        test_mean, test_std, test_movement_direction, test_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(test_loader)
        train_mean, train_std, train_movement_direction, train_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(train_loader)
        summary_lines = [
            f"std_%: {np.round(final_percentage_devation, 3)}",
            f"[TRAIN] mean: {np.round(train_mean, 6)} | std: {np.round(train_std, 6)} | pos_movement: {np.round(train_movement_direction, 3)} | pos_movement_t: {np.round(train_movement_direction_target, 3)}",
            f"[TEST]  mean: {np.round(test_mean, 6)} | std: {np.round(test_std, 6)} | pos_movement: {np.round(test_movement_direction, 3)} | pos_movement_t: {np.round(test_movement_direction_target, 3)}"
        ]
        summary_text = "\n".join(summary_lines)
        console.print(Panel(summary_text, expand=False))
        console.print(Rule("\n TRAIN DATA DISTRIBUTIONS", style="magenta"))
        utils.print_time_and_days_distributions(train_loader)
        console.print(Rule("\n TEST DATA DISTRIBUTIONS", style="magenta"))
        utils.print_time_and_days_distributions(test_loader)

        utils.print_embeddings(model)
        utils.print_feature_attentions(model)

        # add the result to the existing ones
        result_df.loc[len(result_df)] = [
            hidden_size, params, initial_learning_rate, duration, final_train_loss, final_percentage_devation,
            train_std, test_std, train_movement_direction_target, test_movement_direction_target
        ]
        # save the new results
        result_df.to_csv(results_df_path, index=False)


hidden_size, sample_size_stocks = 40, 23

# train (
#     maximal_batch_size=32,
#     hidden_size=hidden_size,
#     initial_learning_rate=0.005,
#     epochs=100,
#     experiment_number=35,
#     dataset=ExponentialWeightsDataset(sample_size_stocks, "stationary_volume_new"),
#     model=ExponentialWeightsModel(hidden_size=hidden_size),
#     results_path = os.path.join("exponential_weights", "outputs")
# )

train (
    maximal_batch_size=32,
    hidden_size=hidden_size,
    initial_learning_rate=0.001,
    epochs=100,
    experiment_number=35,
    dataset=TercileWeightsDataset(sample_size_stocks, "stationary_volume_new"),
    model_factory=lambda: TercileWeightsModel(hidden_size=hidden_size),
    results_path=os.path.join("tercile_weights", "outputs")
)