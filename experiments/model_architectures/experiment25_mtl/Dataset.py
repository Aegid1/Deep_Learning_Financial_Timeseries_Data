import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

def scale_inputs_and_labels(item):
    relevant_features = ["rsi", "macd", "ema", "vwap", "log_volume_change", "eur_usd", "oil_price"]

    stock_prices = np.array(item['features']["log_return"]).reshape(-1, 1)
    stock_prices_tensor = torch.tensor(stock_prices, dtype=torch.float32)

    target = np.array(item['target']).reshape(-1, 1)
    return_target_tensor = torch.tensor(target, dtype=torch.float32)

    # True = 1 = UpwardsMovement, False = 0 = DownwardsMovement
    direction = 1 if item["target"] > 0 else 0

    direction_target_tensor = torch.tensor(direction, dtype=torch.long)

    # only take the last 4 unique oil_prices and currency_rates
    item["features"]["eur_usd"] = [value for i, value in enumerate(item['features']['eur_usd']) if i % 5 == 0][:4]
    item["features"]["oil_price"] = [value for i, value in enumerate(item['features']['oil_price']) if i % 5 == 0][:4]

    features = {
        "stock_prices": stock_prices_tensor
    }

    for key in item["features"].keys():
        if key in relevant_features:
            scaler = MinMaxScaler()
            feature_normalized = scaler.fit_transform(np.array(item["features"][key]).reshape(-1, 1))
            features[key] = torch.tensor(feature_normalized, dtype=torch.float32)
    return features, return_target_tensor, direction_target_tensor


class TechnicalIndicatorsDataSetFeaturesSplit(Dataset):
    def __init__(self, stocks_seq_size):
        self.path = os.path.join('..', '..', 'data', 'samples', 'train_samples', f"sample_size_{stocks_seq_size}_stationary_volume_fixed")
        companies = os.listdir(self.path)
        self.paths = []
        for company in companies:
            self.paths = self.paths + [os.path.join(self.path, company, p) for p in os.listdir(os.path.join(self.path, company)) if p.endswith('.pkl')]
        random.shuffle(self.paths)

        self.stocks_seq_size = stocks_seq_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        features_tensor, return_target_tensor, direction_target_tensor = scale_inputs_and_labels(item)
        return features_tensor, return_target_tensor, direction_target_tensor