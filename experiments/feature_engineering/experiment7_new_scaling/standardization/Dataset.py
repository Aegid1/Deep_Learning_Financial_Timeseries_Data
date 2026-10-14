import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

FEATURES_TO_STANDARDIZE = ["log_return", "rsi", "macdhist", "vwap_p", "log_volume_change", "ema_p"]
CLIP_PERCENTILE = 0.5  # outermost 0.5% of the data (0.25% per side), as in experiment 2
STATS_PATH = os.path.join("standardization", "features_stats.json")


def _scale_inputs_and_labels(item):
    time_mask = np.array(item['features']["time_mask"]).reshape(-1, 1)
    time_mask_tensor = torch.tensor(time_mask, dtype=torch.float32)

    features = { "time_mask": time_mask_tensor }

    with open(STATS_PATH, "r") as file:
        stats = json.load(file)

    for key in item["features"].keys():
        if key in FEATURES_TO_STANDARDIZE:
            feature_scaled = scale_features(
                data=np.array(item["features"][key]).reshape(-1, 1),
                feature_name=key,
                stats=stats[key]
            )
            features[key] = torch.tensor(feature_scaled, dtype=torch.float32)

    target = np.array(item['targets']["log_return"])
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1) # [batch_size, 1]
    return features, target_tensor, str(item["datetime"])


def transform_feature(data, feature_name:str):
    """Maps a raw feature onto a roughly zero-centered scale; clipping and std scaling operate on this scale."""
    if feature_name == "rsi":
        return (data - 50) / 50
    elif feature_name in ("vwap_p", "ema_p"):
        return np.log(data)
    else:
        return data


def compute_feature_statistics(feature_name:str, raw_values):
    """Clip bounds and std of a feature, both measured on the transformed scale, std after clipping."""
    transformed = transform_feature(np.asarray(raw_values, dtype=np.float64), feature_name)
    lower = np.percentile(transformed, CLIP_PERCENTILE / 2)
    upper = np.percentile(transformed, 100 - CLIP_PERCENTILE / 2)
    std = np.std(np.clip(transformed, lower, upper))
    return {"lower": float(lower), "upper": float(upper), "std": float(std)}


def scale_features(data, feature_name:str, stats:dict):
    if feature_name not in FEATURES_TO_STANDARDIZE:
        print(f"feature: {feature_name} is currently not considered to be transformed")
        return data

    transformed = transform_feature(data, feature_name)
    clipped = np.clip(transformed, stats["lower"], stats["upper"])
    return clipped / stats["std"]


class Dataset(Dataset):
    def __init__(self, stocks_seq_size, train_sample_folder):
        self.path = os.path.join('..', '..', '..', 'data', 'samples', 'train_samples', train_sample_folder)
        companies = os.listdir(self.path)
        self.paths = []
        for company in companies:
            paths_company = [os.path.join(self.path, company, p) for p in os.listdir(os.path.join(self.path, company)) if p.endswith('.pkl')]
            self.paths = self.paths + paths_company
        random.shuffle(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index, compute_stds=False):
        with open(self.paths[index], 'rb') as f:
            item = pickle.load(f)

        # checks whether the stds should be computed for train samples of the dataset or the samples should be scaled
        if compute_stds:
            return item
        else:
            features, target_tensor, datetime = _scale_inputs_and_labels(item)
            return features, target_tensor, datetime