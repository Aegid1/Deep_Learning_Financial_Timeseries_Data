import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

class Dataset(Dataset):
    def __init__(self, stocks_seq_size, samples_name:str):
        self.path = os.path.join('..', '..', "..", 'data', "samples", "leftover_data", samples_name)
        self.samples_name = samples_name
        self.stocks_seq_size = stocks_seq_size
        self.paths = os.listdir(self.path)
        self.paths = [p for p in self.paths if p.endswith('.pkl')]
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(os.path.join(self.path, self.paths[idx]), 'rb') as f:
            item = pickle.load(f)

        features_lstms, features = [], []
        relevant_features = ["stock_prices", "rsi", "macd", "vwap", "ema"]
        for feature_name, feature_value in item['features'].items():
            if feature_name in relevant_features:
                scaler = MinMaxScaler()
                feature_normalized = scaler.fit_transform(np.array(feature_value).reshape(-1, 1))
                feature_tensor = torch.tensor(feature_normalized, dtype=torch.float32)
                if feature_name == "ema":
                    features.append(feature_tensor)
                else:
                    features_lstms.append(feature_tensor)

        features = {
            "features_lstm": torch.tensor(np.array(features_lstms),dtype=torch.float32),
            "features": torch.tensor(np.array(features),dtype=torch.float32)
        }
        target = np.array(item['target'])
        target_tensor = torch.tensor(np.log(target), dtype=torch.float32).unsqueeze(-1)  # [batch_size, 1]
        return features, target_tensor, {}