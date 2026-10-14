import os
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_distributions_of_features(samples_folder_name:str, desired_features:list):
    # gather the data of the training samples inside dictionaries containing the features
    path = os.path.join('..', "..", "..", "data", "samples", "train_samples", f"{samples_folder_name}")
    list_of_dicts = []
    company_names = os.listdir(path)
    for company_name in company_names:
        path_with_company = os.path.join(path, company_name)
        pickle_paths = os.listdir(path_with_company)
        for pickle_path in pickle_paths:
            with open(os.path.join(path_with_company, pickle_path), 'rb') as f:
                item = pickle.load(f)
                list_of_dicts.append(item["features"])

    final_dict = {
        key: np.concatenate([d[key] for d in list_of_dicts])
        for key in list_of_dicts[0].keys()
    }
    final_dict_filtered = {key: value for key, value in final_dict.items() if key in desired_features}
    final_dict_filtered = clip_features(0.5, final_dict_filtered, desired_features) # clips outermost 0.5% of data

    fig, axes = plt.subplots(int(len(desired_features)/2), 2, figsize=(15, 10))
    axes = axes.flatten()

    for i, (feature_name, feature) in enumerate(final_dict_filtered.items()):
        transformed_feature = scale_features_accordingly(feature, feature_name)
        mean, std = np.round(np.mean(transformed_feature),4), np.round(np.std(transformed_feature),4)

        # bins = "auto" determines the amount and width of the bins automatically
        sns.histplot(transformed_feature, bins='auto', kde=True, ax=axes[i], color='#1f77b4')

        # computes the amount of bins
        bin_edges = np.histogram_bin_edges(transformed_feature, bins='auto')
        num_bins = len(bin_edges) - 1

        axes[i].set_title(f"{feature_name} | mean: {mean} | std: {std} \n({num_bins} bins automatically generated)")
        axes[i].set_xlabel("Values")
        axes[i].set_ylabel("Distribution")

    plt.tight_layout()
    plt.savefig("feature_distributions.png")


def scale_features_accordingly(data, feature_name:str):
    if feature_name == "log_return":
        return scale_by_std(data)
    elif feature_name == "log_volume_change":
        return scale_by_std(data)
    elif feature_name == "rsi":
        return scale_rsi(data)
    elif feature_name == "macdhist":
        return scale_by_std(data)
    elif feature_name == "vwap_p":
        return scale_by_log_and_std(data)
    elif feature_name == "ema_p":
        return scale_by_log_and_std(data)
    else:
        print(f"feature: {feature_name} is currently not considered to be transformed")
        return data


def scale_rsi(data):
    data = (data - 50) / 50
    return data / np.std(data)

def scale_by_std(data):
    return data / np.std(data)

def scale_by_log_and_std(data):
    log_data = np.log(data)
    return log_data / np.std(log_data)


def plot_values_with_standard_deviation(company_name:str, std:float, experiment:int):
    df = pd.read_csv(os.path.join("..", "..", "data", "stocks", f"{company_name}.csv"), parse_dates=['Datetime'])
    plt.figure(figsize=(12, 10))
    lower_boundary = df["Open"] * (1-std)
    upper_boundary = df["Open"] * (1+std)

    plt.plot(df["Datetime"][:100], df["Open"][:100], label="actual values", color="black")
    plt.plot(df["Datetime"][:100], lower_boundary[:100], label="Lower Boundary", color="red")
    plt.plot(df["Datetime"][:100], upper_boundary[:100], label="Upper Boundary", color="green")

    plt.xlabel("Date")
    plt.ylabel("Open")
    plt.title("std_%-datetime")
    save_path = os.path.join("outputs", f"model_{experiment}", f"std_%_plotted_{company_name}.png")
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()


def standardize_features(features: dict, features_to_scale:list):
    for key, value in features.items():
        if key in features_to_scale:
            value -= np.mean(value)
            features[key] = value / np.std(value)

    return features


def clip_features(percentile, features, features_to_clip):
    for key, value in features.items():
        if key in features_to_clip:
            lower_bound = np.percentile(value, percentile/2)
            upper_bound = np.percentile(value, 100 - percentile/2)
            features[key] = np.clip(value, lower_bound, upper_bound)

    return features


def compute_oil_change(samples_folder_name:str):
    # gather the dictionaries containing the features
    path = os.path.join('..', "samples", "train_samples", f"{samples_folder_name}")
    list_of_dicts = []
    company_names = os.listdir(path)
    for company_name in company_names:
        path_with_company = os.path.join(path, company_name)
        pickle_paths = os.listdir(path_with_company)
        for pickle_path in pickle_paths:
            with open(os.path.join(path_with_company, pickle_path), 'rb') as f:
                item = pickle.load(f)
                list_of_dicts.append(item["features"])

    final_dict = {
        key: np.concatenate([d[key] for d in list_of_dicts])
        for key in list_of_dicts[0].keys()
    }

    oil_price_shifted = final_dict["oil_price"][5:]
    oil_price_change = oil_price_shifted / final_dict["oil_price"][:-5]
    log_oil_price_change = np.log(oil_price_change)

    max_abs_value = np.max(np.abs(log_oil_price_change))
    scaled_oil_change = log_oil_price_change / (max_abs_value)

    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    sns.histplot(scaled_oil_change, bins='auto', kde=True, ax=axes, color='#1f77b4')

    # 0.25% left and 99.75% right outliers
    lower_bound = np.percentile(scaled_oil_change, 0.25)
    upper_bound = np.percentile(scaled_oil_change, 99.75)

    axes.axvline(lower_bound, color='red', linestyle='--', linewidth=1.5, label='0.25% / 99.75% Grenze')
    axes.axvline(upper_bound, color='red', linestyle='--', linewidth=1.5)

    # computes the amount of bins
    bin_edges = np.histogram_bin_edges(scaled_oil_change, bins='auto')
    num_bins = len(bin_edges) - 1

    axes.set_title(f"oil_price_change\n({num_bins} bins automatically generated)")
    axes.set_xlabel("Values")
    axes.set_ylabel("Amount")

    plt.tight_layout()
    plt.savefig("oil_change_distribution.png")


def compute_currency_change(samples_folder_name:str):
    # gather the dictionaries containing the features
    path = os.path.join('..', "samples", "train_samples", f"{samples_folder_name}")
    list_of_dicts = []
    company_names = os.listdir(path)
    for company_name in company_names:
        path_with_company = os.path.join(path, company_name)
        pickle_paths = os.listdir(path_with_company)
        for pickle_path in pickle_paths:
            with open(os.path.join(path_with_company, pickle_path), 'rb') as f:
                item = pickle.load(f)
                list_of_dicts.append(item["features"])

    final_dict = {
        key: np.concatenate([d[key] for d in list_of_dicts])
        for key in list_of_dicts[0].keys()
    }

    currency_shifted = final_dict["eur_usd"][5:]

    currency_change = currency_shifted / final_dict["eur_usd"][:-5]
    log_currency_change = np.log(currency_change)

    max_abs_value = np.max(np.abs(log_currency_change))
    scaled_currency_change = log_currency_change / (max_abs_value)

    fig, axes = plt.subplots(1, 1, figsize=(15, 10))
    sns.histplot(scaled_currency_change, bins='auto', kde=True, ax=axes, color='#1f77b4')

    # 0.25% left and 99.75% right outliers
    lower_bound = np.percentile(scaled_currency_change, 0.25)
    upper_bound = np.percentile(scaled_currency_change, 99.75)

    axes.axvline(lower_bound, color='red', linestyle='--', linewidth=1.5, label='0.25% / 99.75% Grenze')
    axes.axvline(upper_bound, color='red', linestyle='--', linewidth=1.5)

    # computes the amount of bins
    bin_edges = np.histogram_bin_edges(scaled_currency_change, bins='auto')
    num_bins = len(bin_edges) - 1

    axes.set_title(f"currency_change\n({num_bins} bins automatically generated)")
    axes.set_xlabel("Values")
    axes.set_ylabel("Amount")

    plt.tight_layout()
    plt.savefig("currency_change_distribution.png")


def plot_distributions_of_standardized_features(samples_folder_name:str, desired_features:list):
    # gather the data of the training samples inside dictionaries containing the features
    path = os.path.join('..', "..", "..", "data", "samples", "train_samples", f"{samples_folder_name}")
    list_of_dicts = []
    company_names = os.listdir(path)
    for company_name in company_names:
        path_with_company = os.path.join(path, company_name)
        pickle_paths = os.listdir(path_with_company)
        for pickle_path in pickle_paths:
            with open(os.path.join(path_with_company, pickle_path), 'rb') as f:
                item = pickle.load(f)
                list_of_dicts.append(item["features"])

    final_dict = {
        key: np.concatenate([d[key] for d in list_of_dicts])
        for key in list_of_dicts[0].keys()
    }

    final_dict = clip_features(1.5, final_dict, ["macd", "log_return"])
    final_dict = standardize_features(final_dict, ["rsi", "log_return", "macd", "log_volume_change", "vwap", "ema", "eur_usd", "oil_price"])

    features = [value for key, value in final_dict.items() if key in desired_features]
    means_of_features = [np.round(np.mean(value),2) for key, value in final_dict.items() if key in desired_features]
    stds_of_features = [np.round(np.std(value), 2) for key, value in final_dict.items() if key in desired_features]

    fig, axes = plt.subplots(int(len(desired_features)/2), 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, data in enumerate(features):
        # bins = "auto" determines the amount and width of the bins automatically
        sns.histplot(data, bins='auto', kde=True, ax=axes[i], color='#1f77b4')

        lower_bound = np.percentile(data, 0.25)
        upper_bound = np.percentile(data, 99.75)
        axes[i].axvline(lower_bound, color='red', linestyle='--', linewidth=1.5, label='0.25% / 99.75% Grenze')
        axes[i].axvline(upper_bound, color='red', linestyle='--', linewidth=1.5)

        # computes the amount of bins
        bin_edges = np.histogram_bin_edges(data, bins='auto')
        num_bins = len(bin_edges) - 1

        axes[i].set_title(f"{desired_features[i]} mean: {means_of_features[i]} std: {stds_of_features[i]} \n({num_bins} bins automatically generated)")
        axes[i].set_xlabel("Values")
        axes[i].set_ylabel("Amount")

    plt.tight_layout()
    plt.savefig("feature_distributions_standardized.png")


# plot_training_sample_time_distribution("sample_size_23_stationary_volume_fixed")
# plot_distributions_of_features("sample_size_23_stationary_volume_fixed", ["log_return", "log_volume_change", "rsi", "vwap", "ema", "eur_usd", "oil_price", "macd"])
# plot_distributions_of_features_updated("sample_size_23_features_normalized", ["log_return", "log_volume_change", "rsi", "vwap", "ema", "currency_rate_change", "oil_price_change", "macd"])
# plot_distributions_of_features_updated("sample_size_23_stationary_volume_fixed", ["log_return", "log_volume_change", "rsi", "vwap", "ema", "eur_usd", "oil_price", "macd"])
# compute_oil_change("sample_size_23_stationary_volume_fixed")

# compute_currency_change("sample_size_23_stationary_volume_fixed")

#plot_values_with_standard_deviation(company_name="BMW", std = 0.0051, experiment=10)

# plot_distributions_of_features(
#     samples_folder_name="sample_size_23_stationary_volume_new",
#     desired_features=["log_return", "log_volume_change", "rsi", "vwap_p", "ema_p", "macdhist"]
# )
