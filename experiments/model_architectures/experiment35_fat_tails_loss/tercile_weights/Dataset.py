import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

def _scale_inputs_and_labels(item):
    relevant_features_lstms = ["rsi", "macd", "vwap", "log_volume_change", "ema"]

    stock_log_return = np.array(item['features']["log_return"]).reshape(-1, 1)
    stock_log_return_tensor = torch.tensor(stock_log_return, dtype=torch.float32)

    time_mask = np.array(item['features']["time_mask"]).reshape(-1, 1)
    time_mask_tensor = torch.tensor(time_mask, dtype=torch.float32)

    features = { "log_return": stock_log_return_tensor, "time_mask": time_mask_tensor }

    for key in item["features"].keys():
        if key in relevant_features_lstms:
            scaler = MinMaxScaler()
            feature_normalized_2d = scaler.fit_transform(np.array(item["features"][key]).reshape(-1, 1))
            features[key] = torch.tensor(feature_normalized_2d, dtype=torch.float32)

    target = np.array(item['targets']["log_return"])
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1) # [batch_size, 1]
    return features, target_tensor, str(item["datetime"])


def _read_target(path):
    with open(path, "rb") as f:
        item = pickle.load(f)
    # t = np.asarray(item["targets"]["log_return"], dtype=np.float64).ravel()
    # return t[0]
    return item["targets"]["log_return"]


def tercile_weights(y, returns_sorted, W=4, tau=0.05):
    # searches for every element in y the amount of values in returns_sorted that are smaller or equal than the element
    # thus, it provides by the division of the length of returns_sorted the cumulative density function
    u = np.searchsorted(returns_sorted, y, side="right") / len(returns_sorted)
    d = np.abs(u - 0.5)
    return 1 + (W - 1) / (1 + np.exp(-(d - 1 / 6) / tau)) # stronger weights to outer 33% left/right of the distribution


class TechnicalIndicatorsDataSetFeaturesSplit(Dataset):
    def __init__(self, stocks_seq_size, train_sample_folder):
        self.path = os.path.join('..', '..', '..', 'data', 'samples', 'train_samples', train_sample_folder)
        companies = os.listdir(self.path)
        self.paths = []
        for company in companies:
            paths_company = [os.path.join(self.path, company, p) for p in os.listdir(os.path.join(self.path, company)) if p.endswith('.pkl')]
            self.paths = self.paths + paths_company
        random.shuffle(self.paths)

        # store for each sample path the company and the return target
        companies = np.array([os.path.basename(os.path.dirname(p)) for p in self.paths])
        targets = np.array([_read_target(p) for p in self.paths])

        sorted_returns_by_company = {c: np.sort(targets[companies == c]) for c in np.unique(companies)}

        # calculate the weights for a given company and only adjust the indices that correspond to the company
        self.weights = np.empty(len(self.paths), dtype=np.float32)
        for company in np.unique(companies):
            company_mask = companies == company
            weights = tercile_weights(targets[company_mask], sorted_returns_by_company[company])
            self.weights[company_mask] = weights
            self.weights /= weights.sum() # normalize the weights to be in the interval of (0,1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        features, target_tensor, datetime = _scale_inputs_and_labels(item)
        # add weight to the features dict to be able to retrieve it later in the loss calculation
        features["sample_weight"] = torch.tensor([self.weights[idx]], dtype=torch.float32)
        return features, target_tensor, datetime
