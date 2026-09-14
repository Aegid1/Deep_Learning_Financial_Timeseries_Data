import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

def scale_inputs_and_labels(item):
    relevant_features = ["rsi", "macd", "ema", "vwap", "log_volume_change", "oil_price", "eur_usd"]

    stock_prices = np.array(item['features']["log_return"]).reshape(-1, 1)
    stock_prices_tensor = torch.tensor(stock_prices, dtype=torch.float32)

    time_mask = np.array(item['features']["time_mask"]).reshape(-1, 1)
    time_mask_tensor = torch.tensor(time_mask, dtype=torch.float32)

    target = np.array(item['targets']["log_return"]).reshape(-1, 1)
    target_tensor = torch.tensor(target, dtype=torch.float32)

    features = { "log_return": stock_prices_tensor, "time_mask": time_mask_tensor }

    for key in item["features"].keys():
        if key in relevant_features:
            scaler = MinMaxScaler()
            feature_normalized = scaler.fit_transform(np.array(item["features"][key]).reshape(-1, 1))
            features[key] = torch.tensor(feature_normalized, dtype=torch.float32)
    return features, target_tensor


class TechnicalIndicatorsDataSetFeaturesSplit(Dataset):
    def __init__(self, train_sample_folder):
        self.path = os.path.join('..', '..', 'data', 'samples', 'train_samples', train_sample_folder)
        companies = os.listdir(self.path)
        paths = []
        for company in companies:
            paths = paths + [os.path.join(self.path, company, p) for p in os.listdir(os.path.join(self.path, company)) if p.endswith('.pkl')]
        random.shuffle(paths)
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        features_tensor, target_tensor = scale_inputs_and_labels(item)
        return features_tensor, target_tensor
