import os

import numpy as np
import torch
from torch import nn

from test_model import test_model_log_returns
import utils.experiment_utils as utils
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import ModelLessLSTMs as Model

def train(sample_size_stocks:int, maximal_batch_size:int, hidden_size:int, initial_learning_rate:float, epochs:int, experiment_number: int):
    result_df = utils.create_results_csv()
    dataset = TechnicalIndicatorsDataSetFeaturesSplit(sample_size_stocks, "sample_size_23_features_normalized_stocks_stationary_volume_updated")
    final_percentage_devation = np.inf

    for i in range(10):
        model = Model(input_size=1, hidden_size=hidden_size)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) # calculates total amount of params
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-2)

        final_train_loss, final_test_loss, final_percentage_devation_new, duration, model = utils.train_model(model, dataset, maximal_batch_size, epochs, experiment_number, criterion = nn.MSELoss(), optimizer=optimizer)

        print(f"------------------------------------------------\n\n {i} \n std_%: {final_percentage_devation_new}")
        if final_percentage_devation_new < final_percentage_devation:  # only save the model when the std was reduced compared to previous model
            torch.save(model.state_dict(), os.path.join("outputs", f"trained_model_experiment{experiment_number}_{final_percentage_devation_new}.pth"))
            final_percentage_devation = final_percentage_devation_new

        result_df.loc[len(result_df)] = [hidden_size, params, initial_learning_rate, duration, final_train_loss, final_test_loss, final_percentage_devation_new]

    result_df.to_csv(os.path.join("outputs", "results.csv"), index=False)

# TODO initialisierung um 0.81 prozentuale abweichung resultiert in weitaus geringere prozentuale abweichung am Ende ~ 0.45/0.46
# TODO -> Initialisierung mal checken hinsichtlich optimierungspotenzial

train (
    sample_size_stocks=23,
    maximal_batch_size=16,
    hidden_size=50,
    initial_learning_rate=0.001,
    epochs=100,
    experiment_number=28
)