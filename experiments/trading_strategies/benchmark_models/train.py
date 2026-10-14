import os
import torch
from rich.console import Console
from torch import nn
from torch.nn import MSELoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
import utils.experiment_utils as utils
from ALSTM.ALSTM import ALSTMModel
from GAT.GAT import GATM
from GRU.GRU import GRUModel
from Localformer.Localformer import Transformer as Localformer
from TCN.TCN import TCNModel
from Transformer.Transformer import Transformer
from _optuna_optimization.stationary_volume_new.Dataset import TechnicalIndicatorsDataSetFeaturesSplit

console = Console(width=90)


def train(
        maximal_batch_size:int,
        model_params:dict,
        initial_learning_rate:float,
        epochs:int,
        model_name:str,
        model,
        feature_set_name:str,
        dataset:Dataset
):
    # check whether the path for saving the training results already exists
    path = os.path.join("..", "trained_benchmark_models", feature_set_name, model_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    # determine columns of results.csv
    param_columns = list(model_params.keys())
    metric_columns = ["params", "initial_learning_rate", "duration", "final_train_loss", "final_test_loss"]
    all_columns = param_columns + metric_columns

    result_df = utils.create_results_csv(columns=all_columns, path=os.path.join(path, "results.csv"))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) # calculates total amount of params

    optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-4)
    dynamic_eta_min = initial_learning_rate * 0.05
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=dynamic_eta_min)
    # criterion =  nn.SmoothL1Loss(beta=3.5e-3)
    criterion = MSELoss()
    train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)
    final_train_loss, final_test_loss, final_percentage_devation_new, duration, model = utils.train_model(
        model, epochs, criterion, optimizer, train_loader, test_loader, scheduler
    )

    # save the training results and the trained model weights
    torch.save(model.state_dict(), os.path.join(path, f"{final_percentage_devation_new}.pth"))

    param_values = list(model_params.values())
    metric_values = [params, initial_learning_rate, duration, final_train_loss, final_percentage_devation_new]

    result_df.loc[len(result_df)] = param_values + metric_values
    result_df.to_csv(os.path.join(path, "results.csv"), index=False)


for i in range(10):
    current_model_params = {
        "d_feat": 7,
        "hidden_size": 32,
        "num_layers": 4,
        "dropout": 0.2
    }
    train(
        maximal_batch_size=32,
        model_params=current_model_params,
        initial_learning_rate=1e-3,
        epochs=100,
        model_name="ALSTM",
        model=ALSTMModel(**current_model_params),
        feature_set_name="stationary_volume_new",
        dataset=TechnicalIndicatorsDataSetFeaturesSplit(
            23,
            os.path.join("..", '..', '..', 'data', 'samples', 'train_samples', "stationary_volume_new")
        )
    )