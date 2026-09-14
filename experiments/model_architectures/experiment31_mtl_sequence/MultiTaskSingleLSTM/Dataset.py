import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

def scale_inputs_and_labels(item):
    stock_log_return = np.array(item['features']["log_return"]).reshape(-1, 1)

    stock_log_return_tensor = torch.tensor(stock_log_return, dtype=torch.float32)

    time_mask = np.array(item['features']["time_mask"]).reshape(-1, 1)
    time_mask_tensor = torch.tensor(time_mask, dtype=torch.float32)

    features = { "log_return": stock_log_return_tensor, "time_mask": time_mask_tensor}

    target = np.array(item['targets']["log_return"])
    target_tensor = torch.tensor(target, dtype=torch.float32)
    return features, target_tensor, str(item["datetime"])


class TechnicalIndicatorsDataSetFeaturesSplit(Dataset):
    def __init__(self, stocks_seq_size, train_sample_folder):
        self.path = os.path.join("..", '..', '..', 'data', 'samples', 'train_samples', train_sample_folder)
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
        features_tensor, target_tensor, datetime = scale_inputs_and_labels(item)
        return features_tensor, target_tensor, datetime