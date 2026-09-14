import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from data.analysis.analysis import clip_features


def scale_inputs_and_labels(item, means, stds):
    features = { }
    for key in means.keys():
        val = (item["features"][key] - means[key]) / stds[key]
        features[key] = torch.tensor(val, dtype=torch.float32).reshape(-1, 1)

    target = np.array(item['targets']["log_return"]).reshape(-1, 1)
    target_tensor = torch.tensor(target, dtype=torch.float32)

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
        self.means_dict, self.std_dict = { }, { }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with open(self.paths[idx], 'rb') as f:
            item = pickle.load(f)
        features_tensor, target_tensor = scale_inputs_and_labels(item, self.means_dict, self.std_dict)
        return features_tensor, target_tensor

    def calculate_feature_mean_std(self, train_loader):
        features_to_scale = ["rsi", "log_return", "macd", "log_volume_change", "vwap", "ema", "eur_usd", "oil_price"]
        train_sample_indices = train_loader.dataset.indices
        train_sample_paths = [self.paths[i] for i in train_sample_indices]
        list_of_dicts = []
        for pickle_path in train_sample_paths:
            with open(pickle_path, 'rb') as f:
                item = pickle.load(f)
                list_of_dicts.append(item["features"])

        feature_dict = {
            key: np.concatenate([d[key] for d in list_of_dicts])
            for key in list_of_dicts[0].keys()
        }

        feature_dict = clip_features(1.5, feature_dict, ["macd", "log_return"])
        self.means_dict = { key: np.mean(feature_dict[key]) for key in feature_dict.keys() if key in features_to_scale }
        self.std_dict = { key: np.std(feature_dict[key]) for key in feature_dict.keys() if key in features_to_scale }
