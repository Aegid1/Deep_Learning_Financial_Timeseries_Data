import os
import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import pickle

class Dataset(Dataset):
    def __init__(self, stocks_seq_size, samples_name:str):
        self.path = os.path.join('..', '..', "..", 'data', "samples", "train_samples", samples_name)
        self.samples_name = samples_name
        self.stocks_seq_size = stocks_seq_size

        companies = os.listdir(self.path)
        self.paths = []
        for company in companies:
            company_dir = os.path.join(self.path, company)
            if company != "BMW":
                self.paths = self.paths + [os.path.join(company_dir, p) for p in os.listdir(company_dir) if p.endswith('.pkl')]
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)

        features_lstms, features = [], []
        relevant_features = ["stock_prices", "rsi", "ema", "macd", "vwap", "short_term_impact", "long_term_impact"]
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
            "features_lstm": torch.tensor(np.array(features_lstms), dtype=torch.float32),
            "features": torch.tensor(np.array(features), dtype=torch.float32)
        }
        target = np.array(item['target'])
        target_tensor = torch.tensor(np.log(target), dtype=torch.float32).unsqueeze(-1)  # [batch_size, 1]
        return features, target_tensor, {}