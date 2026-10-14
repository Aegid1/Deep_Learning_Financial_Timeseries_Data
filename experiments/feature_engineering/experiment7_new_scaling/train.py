import json
import os
from types import SimpleNamespace
from Model import SimpleLSTM
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils
from min_max_scaling.Dataset import Dataset as DatasetMinMaxScaling
from standardization.Dataset import Dataset as DatasetStandardization
from standardization.Dataset import scale_features, compute_feature_statistics, FEATURES_TO_STANDARDIZE, STATS_PATH

def train(
        maximal_batch_size:int,
        hidden_size:int,
        initial_learning_rate:float,
        weight_decay:float,
        epochs:int,
        experiment_number: int,
        dataset,
        model_factory,
        results_path
):
    results_df_path = os.path.join(results_path, "results.csv")
    for i in range(10):
        result_df = utils.create_results_csv(
            columns=[
                "hidden_size", "params", "initial_learning_rate", "duration", "final_train_loss",
                "final_test_loss", "std_coefficient"
            ],
            path=results_df_path
        )

        model = model_factory()
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) # calculates total amount of params
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=initial_learning_rate * 0.01) # final lr is 1% of initial lr
        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        # clip bounds and stds must come from the training split only
        if isinstance(dataset, DatasetStandardization):
            calculate_feature_statistics(train_loader)

        final_train_loss, final_test_loss, final_percentage_devation, duration, model = utils.train_model(
            model, epochs, nn.MSELoss(), optimizer, train_loader, test_loader, scheduler
        )
        print(f"std_%: {np.round(final_percentage_devation, 3)}")

        # captures how confidently the model makes its predictions
        std_coefficient = test_model_predictions_std(
            model=model,
            test_data_path=os.path.join("..", "..", "..", "data", "stocks_test_data_benchmark"),
            data_scaling_approach="standardization" if isinstance(dataset, DatasetStandardization) else "min_max"
        )
        print(f"std_coefficient_total: {np.round(std_coefficient, 3)}")
        torch.save(model.state_dict(), os.path.join(results_path, f"trained_model_experiment{experiment_number}_{final_percentage_devation}.pth"))

        # add the result to the existing ones
        result_df.loc[len(result_df)] = [
            hidden_size, params, initial_learning_rate, duration, final_train_loss, final_percentage_devation, std_coefficient
        ]
        result_df.to_csv(results_df_path, index=False)


def test_model_predictions_std(
        model,
        test_data_path:str,
        data_scaling_approach:str
):
    sequence_size, max_batch_size = 23, 32
    predictions_list, targets_list = [], []

    stocks_to_test = [path.replace(".csv", "") for path in os.listdir(test_data_path)]
    for stock in stocks_to_test:
        print(f"processing stock: {stock}")
        data = pd.read_csv(os.path.join(test_data_path, f"{stock}.csv"))

        # simple sliding window
        for i in range(len(data) - sequence_size - 1):
            window = data.iloc[i: i + sequence_size]
            target = data.iloc[i + sequence_size]["log_return"]
            targets_list.append(target)

            # scale the features
            if data_scaling_approach == "standardization":
                scaled_window = scale_data_standardization(window)
            else:
                scaled_window = scale_data_min_max(window)

            prediction = model(**scaled_window)
            predictions_list.append(prediction.item())

    std_predictions = np.std(np.array(predictions_list))
    std_targets = np.std(np.array(targets_list))
    return std_predictions / std_targets


def scale_data_standardization(window: pd.DataFrame):
    features = {}

    with open(STATS_PATH, "r") as f:
        stats = json.load(f)

    time_mask = window["time_mask"].to_numpy().reshape(-1, 1)
    features["time_mask"] = torch.tensor(time_mask, dtype=torch.float32).unsqueeze(0)

    for key in window.columns:
        if key.lower() in FEATURES_TO_STANDARDIZE: # key is always a string, thus the usage of .lower() is safe
            data = window[key].to_numpy().reshape(-1, 1)
            key_lower = key.lower()
            feature_scaled = scale_features(data, key_lower, stats[key_lower])
            features[key_lower] = torch.tensor(feature_scaled, dtype=torch.float32).unsqueeze(0)

    return features


def calculate_feature_statistics(dataloader):
    """Computes clip bounds and std per feature on the training samples only and stores them as .json."""
    feature_values_dict = {name: [] for name in FEATURES_TO_STANDARDIZE}

    # iterate over all indices of training samples
    subset = dataloader.dataset
    dataset = subset.dataset # this is the reference to the original dataset
    for i in range(len(subset)):
        index = subset.indices[i]
        sample = dataset.__getitem__(index, compute_stds=True)

        for name in FEATURES_TO_STANDARDIZE:
            feature_values_dict[name].append(np.asarray(sample["features"][name]).reshape(-1))

    stats = {
        name: compute_feature_statistics(name, np.concatenate(values))
        for name, values in feature_values_dict.items()
    }

    # overwrites an already existing .json from an old run
    with open(STATS_PATH, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

    return stats


def scale_data_min_max(data):
    relevant_features_lstms = ["rsi", "macdhist", "vwap_p", "log_volume_change"]
    features = {}

    stock_log_return = data["log_return"].to_numpy().reshape(-1, 1)
    features["log_return"] = torch.tensor(stock_log_return, dtype=torch.float32).unsqueeze(0)

    time_mask = data["time_mask"].to_numpy().reshape(-1, 1)
    features["time_mask"] = torch.tensor(time_mask, dtype=torch.float32).unsqueeze(0)

    ema = data["EMA_p"].to_numpy().reshape(-1, 1)
    features["ema_p"] = torch.tensor(ema, dtype=torch.float32).unsqueeze(0)

    # scale relevant features with min-max-scaling
    for key in data.columns:
        if key.lower() in relevant_features_lstms:
            scaler = MinMaxScaler()
            data_2d = data[key].to_numpy().reshape(-1, 1)
            key_lower = key.lower()
            feature_normalized = scaler.fit_transform(data_2d)
            features[key_lower] = torch.tensor(feature_normalized, dtype=torch.float32).unsqueeze(0)

    return features


sample_size_stocks = 23

# path = os.path.join("standardization", "param_config.json")
# with open(path, "r", encoding="utf-8") as f:
#     config = json.load(f)["best_params"]
#     params = SimpleNamespace(**config)
#
# train(
#     maximal_batch_size=32,
#     hidden_size=params.hidden_size,
#     initial_learning_rate=params.lr,
#     epochs=400,
#     experiment_number=7,
#     dataset=DatasetStandardization(sample_size_stocks, "sample_size_23_stationary_volume_new"),
#     model_factory=lambda: SimpleLSTM(input_size=7, hidden_size=params.hidden_size, num_layers=params.num_layers, dropout=0.2),
#     results_path=os.path.join("standardization", "outputs"),
#     weight_decay=params.weight_decay
# )

# path = os.path.join("min_max_scaling", "param_config.json")
# with open(path, "r", encoding="utf-8") as f:
#     config = json.load(f)["best_params"]
#     params = SimpleNamespace(**config)
#
# train (
#     maximal_batch_size=32,
#     hidden_size=params.hidden_size,
#     initial_learning_rate=params.lr,
#     epochs=40,
#     experiment_number=7,
#     dataset=DatasetMinMaxScaling(sample_size_stocks, "sample_size_23_stationary_volume_new"),
#     model_factory=lambda: SimpleLSTM(input_size=7, hidden_size=params.hidden_size, num_layers=params.num_layers, dropout=0.2),
#     results_path=os.path.join("min_max_scaling", "outputs"),
#     weight_decay=params.weight_decay
# )