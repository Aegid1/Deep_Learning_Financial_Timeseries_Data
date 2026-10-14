import os
import numpy as np
import pandas as pd
import torch
from experiments.trading_strategies.simulation import benchmark_models
from numpy.lib.stride_tricks import sliding_window_view

def apply_trading_strategy(
        sigma_market,
        sigma_ensemble,
        predictions,
        costs,
        current_weights
):
    alpha = 1.5  # penalty for ensemble uncertainty -> the higher, the more conservative
    w_max = 1.0  # max leverage (1.0 = max 100% of the capital can be used)
    scaling = 2.0 # hyperparam for scaling
    sell_threshold = -0.003  # at this threshold the stock is sold

    # centers the returns around 0, e.g.: 1.05 becomes 0.05 (Long-Signal), 0.98 becomes -0.02 (Short-Signal)
    expected_returns = predictions - 1.0
    net_returns = expected_returns - costs

    total_uncertainty = sigma_market + (alpha * sigma_ensemble)

    # (net_return / uncertainty) * scaling_factor
    raw_buy_weights = (net_returns / total_uncertainty) * scaling
    buy_weights = np.clip(raw_buy_weights, 0.0, w_max)

    # define signals
    buy_signal = net_returns > 0
    sell_signal = expected_returns <= sell_threshold

    target_weights = current_weights.copy()

    # updates the previous weights based on the calculated signals for determining whether to hold, sell or buy
    target_weights = np.where(buy_signal, buy_weights, target_weights)
    target_weights = np.where(sell_signal, 0.0, target_weights)

    return target_weights


def create_scaled_batches(df: pd.DataFrame, sequence_size: int):
    """
    Assumes that the data is provided as a dataframe, where the features are the columns
    and the data is sorted in chronological order
    """
    df.columns = df.columns.str.lower() # make col names lowercase to ensure they can be found
    # relevant_features = ["rsi", "log_volume_change", "vwap", "macd", "ema"]
    relevant_features = ["rsi", "log_volume_change", "vwap", "macd"]
    features_dict = {}
    num_windows = len(df) - sequence_size
    target_dates = df.index[sequence_size:]

    # the last window is not used because there is no target available in the dataset
    log_return_window = sliding_window_view(df["log_return"].values, sequence_size)[:num_windows]
    time_mask_window = sliding_window_view(df["time_mask"].values, sequence_size)[:num_windows]
    ema_mask_window = sliding_window_view(df["ema"].values, sequence_size)[:num_windows]

    features_dict["log_return"] = torch.tensor(log_return_window, dtype=torch.float32).unsqueeze(-1)
    features_dict["time_mask"] = torch.tensor(time_mask_window, dtype=torch.float32).unsqueeze(-1)
    features_dict["ema"] = torch.tensor(ema_mask_window, dtype=torch.float32).unsqueeze(-1)

    for feat in relevant_features:
        if feat in df.columns:
            feat_win = sliding_window_view(df[feat].values, sequence_size)[:num_windows]

            # determine min and max for every window (at axis=1) in order to normalize the data
            w_min, w_max = feat_win.min(axis=1, keepdims=True), feat_win.max(axis=1, keepdims=True)
            denominator = w_max - w_min
            denominator[denominator == 0] = 1e-8 # prevent division by zero
            scaled_win = (feat_win - w_min) / denominator
            features_dict[feat] = torch.tensor(scaled_win, dtype=torch.float32).unsqueeze(-1)

    return features_dict, target_dates


FEATURE_SET_NAME = "stationary_volume_new"  # must match the feature_set_name on which the models are trained on

benchmark_models(
    # models = ["ALSTM", "GRU", "GAT"],
    # models = ["MILSTM_sample_reweighting_less_weight_decay"],
    models = ["ALSTM"],
    models_path=os.path.join("..", "trained_benchmark_models", FEATURE_SET_NAME),
    test_data_path=os.path.join("..", "..", "..", "data", "stocks_test_data_benchmark"),
    stocks_to_benchmark=[
        "Airbus", "Allianz", "BMW", "Deutsche Bank", "Deutsche Telekom", "Mercedes-Benz", "Porsche", "SAP",
        "Siemens", "Siemens Healthineers", "Volkswagen"
    ],
    output_path="outputs",
    scale_inputs=create_scaled_batches,
    trading_strategy=apply_trading_strategy
)