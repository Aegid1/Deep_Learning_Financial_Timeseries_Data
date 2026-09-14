import os
import numpy as np
import torch
from torch import nn
import utils.experiment_utils as utils
from torch.optim.lr_scheduler import CosineAnnealingLR
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import ModelLessLSTMs as Model

def train(maximal_batch_size:int, hidden_size:int, initial_learning_rate:float, epochs:int, experiment_number: int):
    result_df = utils.create_results_csv()
    dataset = TechnicalIndicatorsDataSetFeaturesSplit("sample_size_23_features_normalized_stocks_stationary_volume_updated")

    final_percentage_devation = np.inf
    for i in range(10):
        model = Model(hidden_size=hidden_size)
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        final_train_loss, final_test_loss, final_percentage_devation_new, duration, model = utils.train_model(
            model, epochs, nn.SmoothL1Loss(beta=4e-3), optimizer, train_loader, test_loader, scheduler
        )

        if final_percentage_devation_new < final_percentage_devation: # only save the model when the std was reduced compared to previous model
            torch.save(model.state_dict(), os.path.join("outputs", f"trained_model_experiment{experiment_number}_{final_percentage_devation_new}.pth"))
            final_percentage_devation = final_percentage_devation_new

        test_mean, test_std, test_movement_direction, test_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(test_loader)
        train_mean, train_std, train_movement_direction, train_movement_direction_target = utils.compute_dataloader_mean_std_movement_direction(train_loader)
        print(f"------------------------------------------------\n\n {i} \n std_%: {final_percentage_devation_new}")
        print(f"[TRAIN] mean: {train_mean} | std: {train_std} | pos_movements: {train_movement_direction} | pos_movements_target: {train_movement_direction_target}")
        print(f"[TEST] mean: {test_mean} | std: {test_std} | pos_movements: {test_movement_direction} | pos_movements_target: {test_movement_direction_target}")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) # calculates total amount of params
        result_df.loc[len(result_df)] = [hidden_size, params, initial_learning_rate, duration, final_train_loss, final_test_loss, final_percentage_devation_new]

    result_df.to_csv(os.path.join("outputs", "results.csv"), index=False)


train (
    maximal_batch_size=16,
    hidden_size=50,
    initial_learning_rate=0.0004,
    epochs=100,
    experiment_number=31
)