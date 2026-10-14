import json
import os
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from Model import SimpleLSTM
from train import scale_data_standardization

# Naive baseline: always predicts a log-return of 0 (= no price change), regardless of the input.
# The deviation is computed like this:
# mean(|exp(prediction) - exp(target)|) * 100, here with prediction = 0.

SEQUENCE_SIZE = 23
DATA_PATH = os.path.join("..", "..", "..", "data")


def percentage_deviation_of_predictions(predictions: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean(np.abs(np.exp(predictions) - np.exp(targets))) * 100)


def baseline_on_benchmark_test_data(test_data_path: str, sequence_size: int = SEQUENCE_SIZE):
    """Same windows/targets as test_model_predictions_std in train.py: the target is the log_return of the row after each window."""
    all_targets, all_predictions = [], []
    stocks = sorted(path.replace(".csv", "") for path in os.listdir(test_data_path) if path.endswith(".csv"))

    for stock in stocks:
        data = pd.read_csv(os.path.join(test_data_path, f"{stock}.csv"))
        targets = data["log_return"].iloc[sequence_size: len(data) - 1].to_numpy(dtype=np.float64)
        all_targets.append(targets)
        predictions = np.zeros_like(targets)
        all_predictions.append(predictions)

        print(f"{stock}: deviation: {percentage_deviation_of_predictions(predictions, targets):.3f}%")

    total_deviation = percentage_deviation_of_predictions(np.concatenate(all_predictions), np.concatenate(all_targets))
    return total_deviation


def test_model_predictions_std(
        model,
        test_data_path:str
):
    sequence_size, max_batch_size = 23, 32
    predictions_list, targets_list = [], []

    stocks_to_test = [path.replace(".csv", "") for path in os.listdir(test_data_path)]
    for stock in stocks_to_test:
        data = pd.read_csv(os.path.join(test_data_path, f"{stock}.csv"))

        predictions_list_stock, targets_list_stock = [], []
        # simple sliding window
        for i in range(len(data) - sequence_size - 1):
            window = data.iloc[i: i + sequence_size]
            target = data.iloc[i + sequence_size]["log_return"]
            targets_list.append(target)
            targets_list_stock.append(target)

            # scale the features
            scaled_window = scale_data_standardization(window)

            prediction = model(**scaled_window)
            predictions_list.append(prediction.item())
            predictions_list_stock.append(prediction.item())

        predictions_array_stock = np.array(predictions_list_stock)
        targets_array_stock = np.array(targets_list_stock)
        deviation = np.round(percentage_deviation_of_predictions(predictions_array_stock, targets_array_stock), 3)
        std_coefficient = np.std(predictions_array_stock) / np.std(targets_array_stock)

        print(f"{stock} | deviation: {deviation}% | std_coefficient: {std_coefficient}")

    predictions_array, targets_array = np.array(predictions_list), np.array(targets_list)
    std_predictions, std_targets = np.std(np.array(predictions_list)), np.std(np.array(targets_list))

    std_coefficient = np.round(std_predictions / std_targets, 3)
    total_deviation = np.round(percentage_deviation_of_predictions(predictions_array, targets_array), 3)
    return total_deviation, std_coefficient


def compare_baseline_to_standardization_scaled_model(model_params_path:str, test_data_path:str):
    path = os.path.join("standardization", "param_config.json")
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)["best_params"]
        params = SimpleNamespace(**config)

    model = SimpleLSTM(input_size=7, hidden_size=params.hidden_size, num_layers=params.num_layers, dropout=0.2)
    model.load_state_dict(torch.load(model_params_path, weights_only=True))
    model.eval()

    return test_model_predictions_std(model, test_data_path)


total_deviation = baseline_on_benchmark_test_data(os.path.join(DATA_PATH, "stocks_test_data_benchmark"))
print(f"TOTAL percentage deviation of always predicting 0: {total_deviation:.3f}%")

print("-----------------------------------------------------")

total_deviation, std_coefficient = compare_baseline_to_standardization_scaled_model(
    model_params_path=os.path.join("standardization", "outputs", "trained_model_experiment7_0.537.pth"),
    test_data_path=os.path.join(DATA_PATH, "stocks_test_data_benchmark")
)
print(f"TOTAL percentage deviation of model: {total_deviation}% | std_coefficient: {std_coefficient}")
