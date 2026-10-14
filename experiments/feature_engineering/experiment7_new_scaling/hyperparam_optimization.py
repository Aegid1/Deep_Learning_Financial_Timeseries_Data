import json
import os

import numpy as np
import torch
import torch.nn as nn
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from min_max_scaling.Dataset import Dataset as DatasetMinMaxScaling
from standardization.Dataset import Dataset as DatasetStandardization
from train import calculate_feature_statistics
import utils.experiment_utils as utils
from Model import SimpleLSTM

class ModelObjective:
    def __init__(self, dataset, maximal_batch_size, epochs, runs_per_trial=5):
        self.dataset = dataset
        self.maximal_batch_size = maximal_batch_size
        self.epochs = epochs
        self.runs_per_trial = runs_per_trial

    def __call__(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32, 64])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        config_str = f"hidden: {hidden_size} | layers: {num_layers} | lr: {round(lr, 5)} | weight_decay: {round(weight_decay, 5)}"
        print(f"Training models with the parameter config: {config_str}")

        # train multiple models with a given parameter configuration
        deviations = []
        for i in range(self.runs_per_trial):
            print(f"Training model number: {i + 1}")
            train_loader, test_loader = utils.create_train_test_split(self.dataset, self.maximal_batch_size)

            # calculates clip bounds and stds of the features for the standardization approach
            if isinstance(self.dataset, DatasetStandardization):
                calculate_feature_statistics(train_loader)

            model = SimpleLSTM(input_size=7, hidden_size=hidden_size, num_layers=num_layers, dropout=0.2)
            number_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            dynamic_eta_min = lr * 0.01
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=dynamic_eta_min)

            _, _, final_deviation, _, _ = utils.train_model(
                model, self.epochs, nn.MSELoss(), optimizer, train_loader, test_loader, scheduler, False
            )
            print(f"achieved final deviation: {final_deviation}")
            deviations.append(final_deviation)

        trial.set_user_attr("all_deviations", deviations)
        trial.set_user_attr("n_params", number_of_params)
        trial.set_user_attr("config_label", config_str)

        return np.mean(deviations)


def perform_optuna_study(dataset: Dataset, results_path: str, n_trials: int, runs_per_trial:int, epochs:int):
    objective = ModelObjective(
        dataset=dataset,
        maximal_batch_size=32,
        epochs=epochs,
        runs_per_trial=runs_per_trial
    )

    study = optuna.create_study(direction="minimize", study_name=f"hyperparam_search")
    study.optimize(objective, n_trials=n_trials)

    print("best parameters:", study.best_params)
    print("smallest average deviation:", study.best_value)

    config_data = {
        "best_value": study.best_value,
        "best_params": study.best_params
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)

    print(f"Best configuration saved")
    return study


# optuna_study = perform_optuna_study(
#     dataset=DatasetMinMaxScaling(23, "sample_size_23_stationary_volume_new"),
#     results_path=os.path.join("min_max_scaling", "param_config.json"),
#     n_trials=5, # 5
#     runs_per_trial=5, # 5
#     epochs=30 # 30
# )


# optuna_study = perform_optuna_study(
#     dataset=DatasetStandardization(23, "sample_size_23_stationary_volume_new"),
#     results_path=os.path.join("standardization", "param_config.json"),
#     n_trials=5, # 5
#     runs_per_trial=5, # 5
#     epochs=30 # 30
# )