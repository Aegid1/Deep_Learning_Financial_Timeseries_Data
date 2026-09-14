import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils

class ModelObjective:
    def __init__(self, model_builder_func, dataset, maximal_batch_size, epochs, runs_per_trial=5):
        self.model_builder_func = model_builder_func
        self.dataset = dataset
        self.maximal_batch_size = maximal_batch_size
        self.epochs = epochs
        self.runs_per_trial = runs_per_trial

    def __call__(self, trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        deviations = []
        create_model, config_str, number_of_params = self.model_builder_func(trial, self.dataset.amount_of_features)
        config_str += f" | lr: {round(lr, 5)}"
        print(f"Training models with the parameter config: {config_str}")

        # 1. train multiple models with a given parameter configuration
        for i in range(self.runs_per_trial):
            print(f"Training model number: {i+1}")
            train_loader, test_loader = utils.create_train_test_split(self.dataset, self.maximal_batch_size)

            # 2. initialize model with parameter configuration
            model = create_model()
            dynamic_eta_min = lr * 0.05
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=dynamic_eta_min)

            # 3. actually train the model
            _, _, final_deviation, _, _ = utils.train_model(
                model, self.epochs, nn.SmoothL1Loss(beta=3.5e-3), optimizer, train_loader, test_loader, scheduler, False
            )
            print(f"achieved final deviation: {final_deviation}")
            deviations.append(final_deviation)

        # 4. collect metadata for the later visualization of optuna optimization results
        trial.set_user_attr("all_deviations", deviations)
        trial.set_user_attr("n_params", number_of_params)
        trial.set_user_attr("config_label", config_str)

        # optuna should optimize the mean of the final deviations
        return np.mean(deviations)