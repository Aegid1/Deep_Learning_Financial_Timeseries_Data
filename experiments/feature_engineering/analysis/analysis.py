import os
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from experiments.experiment12_model_and_training_data_analysis.test_model import test_model

def create_time_distribution_dataframe(samples_folder_name:str):
    time_distribution_df = pd.DataFrame(columns=["Company", "Day_Name", "Time"])

    path = os.path.join('..', "samples", "train_samples", f"{samples_folder_name}")
    company_names = os.listdir(path)

    for company_name in company_names:
        path_with_company = os.path.join(path, company_name)
        pickle_paths = os.listdir(path_with_company)
        for pickle_path in pickle_paths:
            with open(os.path.join(path_with_company, pickle_path), 'rb') as f:
                item = pickle.load(f)
                day_name = pd.to_datetime(item['datetime']).day_name()
                time = str(item['datetime']).split(" ")[1]
                company_name = item['company_name']
                new_row = {
                    "Company": company_name,
                    "Day_Name": day_name,
                    "Time": time,
                }
                time_distribution_df.loc[len(time_distribution_df)] = new_row

    time_distribution_df.to_csv(os.path.join("time_distribution.csv"))
    return time_distribution_df


def plot_training_sample_time_distribution(samples_folder_name:str):
    if os.path.exists(os.path.join("time_distribution.csv")):
        df = pd.read_csv(os.path.join("time_distribution.csv"))
    else:
        df = create_time_distribution_dataframe(samples_folder_name)

    # creates the barplot for the day distribution
    day_counts_df = df["Day_Name"].value_counts().reset_index()
    total = day_counts_df["count"].sum()
    day_counts_df["relative"] = ((day_counts_df["count"] / total) * 100).round(1)
    plt.bar(day_counts_df["Day_Name"], day_counts_df["count"])

    # adding relative values on top of bars
    for i in range(len(day_counts_df)):
        plt.text(i, day_counts_df["count"].iloc[i], f"{day_counts_df["relative"].iloc[i]} %", ha="center", fontweight="bold")  # Placing text slightly above the bar

    plt.title('Days Distribution')
    plt.xlabel('days')
    plt.ylabel('count')
    plt.savefig(os.path.join("days_distribution.png"), format='png', dpi=300)
    plt.show()

    # creates the barplot for the time distribution
    time_counts_df = df["Time"].value_counts().reset_index()
    for index, row in time_counts_df.iterrows():
        print(row['Time'].split("+")[0][:5])
        time_counts_df.loc[index, "Time"] = row['Time'].split("+")[0][:5]
    plt.bar(time_counts_df["Time"], time_counts_df["count"])

    # adding relative values on top of bars
    time_counts_df["relative"] = ((time_counts_df["count"] / total) * 100).round(1)
    for i in range(len(time_counts_df)):
        plt.text(i, time_counts_df["count"].iloc[i], f"{time_counts_df["relative"].iloc[i]} %", ha="center", fontweight="bold")  # Placing text slightly above the bar

    plt.title('Time Distribution')
    plt.xlabel('times')
    plt.ylabel('count')
    plt.savefig(os.path.join("time_distribution.png"), format='png', dpi=300)
    plt.show()


def create_prediction_difference_distribution(model, experiment:str, sample_size: int):
    df = test_model(model, sample_size, os.path.join("outputs", f"model_{experiment}", f"trained_model_experiment{experiment}.pth"))
    df.to_csv(os.path.join("outputs", f"model_{experiment}", "prediction_errors_distribution.csv"))
    return df


def plot_prediction_difference_distribution(model, experiment:str, sample_size: int):
    if os.path.exists(os.path.join("outputs", f"model_{experiment}", "prediction_errors_distribution.csv")):
        df = pd.read_csv(os.path.join("outputs", f"model_{experiment}","prediction_errors_distribution.csv"))
    else:
        df = create_prediction_difference_distribution(model, experiment, sample_size)

    errors = abs(df["Difference"])
    total_errors = sum(errors.values)

    # overall highest errors at specific days
    total_error_sum_days = df['Difference'].abs().groupby(df['Day']).sum()
    raw_relative_errors = ((total_error_sum_days.values / total_errors) * 100).round(3)
    plt.bar(total_error_sum_days.index, total_error_sum_days.values)
    for i in range(len(total_error_sum_days.index)): # adding relative values on top of bars
        plt.text(i, total_error_sum_days.values[i], f"{raw_relative_errors[i]} %", ha="center", fontweight="bold")

    plt.title('Error distribution on specific days'), plt.xlabel('days'), plt.ylabel('error_sum')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "errors_distribution_days.png"), format='png', dpi=300)
    plt.show()

    # overall highest errors at specific times
    total_error_sum_times = df['Difference'].abs().groupby(df['Time']).sum()
    raw_relative_errors = ((total_error_sum_times.values / total_errors) * 100).round(3)
    plt.bar(total_error_sum_times.index, total_error_sum_times.values)
    for i in range(len(total_error_sum_times.index)):  # adding relative values on top of bars
        plt.text(i, total_error_sum_times.values[i], f"{raw_relative_errors[i]} %", ha="center", fontweight="bold")

    plt.title('Error distribution on specific times'), plt.xlabel('times'), plt.ylabel('error_sum')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "errors_distribution_times.png"), format='png', dpi=300)
    plt.show()

    # average of errors at specific days, sum of errors in relation to how often a prediction was on that day
    day_counts_df = df["Day"].value_counts().reset_index().sort_values(by="Day")
    total_error_sum_days = df['Difference'].abs().groupby(df['Day']).sum()
    raw_errors = abs(total_error_sum_days.values)
    raw_errors_in_relation_to_amount = raw_errors / day_counts_df["count"]
    plt.bar(total_error_sum_days.index, raw_errors_in_relation_to_amount)
    for i in range(len(total_error_sum_days.index)):  # adding relative values on top of bars
        plt.text(i, raw_errors[i], f"{raw_errors_in_relation_to_amount[i]} %", ha="center", fontweight="bold")

    plt.title('Error distribution on specific days'), plt.xlabel('days'), plt.ylabel('average_error')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "average_errors_on_specific_days.png"), format='png', dpi=300)
    plt.show()

    # average of errors at specific days, sum of errors in relation to how often a prediction was on that day
    times_counts_df = df["Time"].value_counts().reset_index().sort_values(by="Time")
    total_error_sum_times = df['Difference'].abs().groupby(df['Time']).sum()
    raw_errors = abs(total_error_sum_times.values)
    raw_errors_in_relation_to_amount = raw_errors / times_counts_df["count"]
    plt.bar(total_error_sum_times.index, raw_errors_in_relation_to_amount)
    for i in range(len(total_error_sum_times.index)):  # adding relative values on top of bars
        plt.text(i, raw_errors[i], f"{raw_errors_in_relation_to_amount[i]} %", ha="center", fontweight="bold")

    plt.title('Error distribution on specific days'), plt.xlabel('days'), plt.ylabel('average_error')
    plt.savefig(os.path.join("outputs", f"model_{experiment}","average_errors_on_specific_times.png"), format='png', dpi=300)
    plt.show()

    # relative amount of correctly predicting the direction at specific days
    day_counts_df = df["Day"].value_counts().reset_index().sort_values(by="Day")
    mask = ( ((df["Actual_Change_Direction"] > 0) & (df["Predicted_Change_Direction"] > 0)) | ((df["Actual_Change_Direction"] <= 0) & (df["Predicted_Change_Direction"] <= 0)) )
    counts_per_day = df[mask].groupby("Day").size()
    relative_amount_correct_directions = ((counts_per_day.values / day_counts_df["count"]) * 100).round(3)
    print(relative_amount_correct_directions)
    plt.bar(counts_per_day.index, relative_amount_correct_directions)
    for i in range(len(counts_per_day.index)):
        plt.text(i, relative_amount_correct_directions.values[i], f"{relative_amount_correct_directions.values[i]} %", ha="center", fontweight="bold")

    plt.title('correct predictions of change on specific days'), plt.xlabel('days'), plt.ylabel('relative_amount_correct_direction')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "correct_direction_on_specific_days.png"), format='png',dpi=300)
    plt.show()

    # relative amount of correctly predicting the direction at specific times
    time_counts_df = df["Time"].value_counts().reset_index()
    mask = (((df["Actual_Change_Direction"] > 0) & (df["Predicted_Change_Direction"] > 0)) | (
                (df["Actual_Change_Direction"] <= 0) & (df["Predicted_Change_Direction"] <= 0)))
    counts_per_time = df[mask].groupby("Time").size()
    relative_amount_correct_directions = ((counts_per_time.values / time_counts_df["count"]) * 100).round(3)
    plt.bar(counts_per_time.index, relative_amount_correct_directions)
    for i in range(len(counts_per_time.index)):
        plt.text(i, relative_amount_correct_directions.values[i], f"{relative_amount_correct_directions.values[i]} %", ha="center", fontweight="bold")

    plt.title('correct predictions of change at specific times'), plt.xlabel('days'), plt.ylabel('relative_amount_correct_direction')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "correct_direction_on_specific_times.png"), format='png', dpi=300)
    plt.show()


def get_stock_up_downward_movements(samples_folder_name:str):
    path = os.path.join('..', "samples", "train_samples", f"{samples_folder_name}")
    company_names = os.listdir(path)
    upwards_movements, total_movements = 0, 0
    for company_name in company_names:
        path_with_company = os.path.join(path, company_name)
        pickle_paths = os.listdir(path_with_company)
        for pickle_path in pickle_paths:
            with open(os.path.join(path_with_company, pickle_path), 'rb') as f:
                item = pickle.load(f)
                stock_log_returns = item["features"]["stock_prices"]
                upwards_movements += len(stock_log_returns[stock_log_returns > 0])
                total_movements += len(stock_log_returns)

    plt.bar(["upwards", "downwards"], [upwards_movements, total_movements - upwards_movements])
    plt.text(0, upwards_movements, f"{upwards_movements/total_movements} %", ha="center", fontweight="bold")  # Placing text slightly above the bar
    plt.text(1, total_movements-upwards_movements, f"{(total_movements-upwards_movements)/total_movements} %", ha="center", fontweight="bold")  # Placing text slightly above the bar

    plt.title('Movements Distribution')
    plt.xlabel('movements')
    plt.ylabel('count')
    plt.savefig(os.path.join("stock_movements_distribution.png"), format='png', dpi=300)
    plt.show()


def plot_distributions_of_features(samples_folder_name:str, desired_features:list):
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

    feature_names = list(final_dict.keys())
    filtered_feature_names = [feature for feature in feature_names if feature in desired_features]

    features = [value for key, value in final_dict.items() if key in filtered_feature_names]
    fig, axes = plt.subplots(int(len(filtered_feature_names)/2), 2, figsize=(15, 10))
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

        axes[i].set_title(f"{filtered_feature_names[i]}\n({num_bins} bins automatically generated)")
        axes[i].set_xlabel("Values")
        axes[i].set_ylabel("Amount")

    plt.tight_layout()
    plt.savefig("feature_distributions.png")


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


def clip_features(clipped_percentile, features, features_to_clip):
    for key, value in features.items():
        if key in features_to_clip:
            lower_bound = np.percentile(value, clipped_percentile/2)
            upper_bound = np.percentile(value, 100 - clipped_percentile/2)
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


def plot_distributions_of_features_updated(samples_folder_name:str, desired_features:list):
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

    final_dict = clip_features(1.5, final_dict, ["macd", "log_return"])
    final_dict = standardize_features(final_dict, ["rsi", "log_return", "macd", "log_volume_change", "vwap", "ema", "eur_usd", "oil_price"])

    feature_names = list(final_dict.keys())
    filtered_feature_names = [feature for feature in feature_names if feature in desired_features]

    features = [value for key, value in final_dict.items() if key in filtered_feature_names]
    means_of_features = [np.round(np.mean(value),2) for key, value in final_dict.items() if key in filtered_feature_names]
    stds_of_features = [np.round(np.std(value), 2) for key, value in final_dict.items() if key in filtered_feature_names]

    fig, axes = plt.subplots(int(len(filtered_feature_names)/2), 2, figsize=(15, 10))
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

        axes[i].set_title(f"{filtered_feature_names[i]} mean: {means_of_features[i]} std: {stds_of_features[i]} \n({num_bins} bins automatically generated)")
        axes[i].set_xlabel("Values")
        axes[i].set_ylabel("Amount")

    plt.tight_layout()
    plt.savefig("feature_distributions_updated.png")


# plot_training_sample_time_distribution("sample_size_23_stationary_volume_fixed")
# get_stock_up_downward_movements("sample_size_23_stationary_volume_fixed")
# plot_distributions_of_features("sample_size_23_stationary_volume_fixed", ["log_return", "log_volume_change", "rsi", "vwap", "ema", "eur_usd", "oil_price", "macd"])
# plot_distributions_of_features_updated("sample_size_23_features_normalized", ["log_return", "log_volume_change", "rsi", "vwap", "ema", "currency_rate_change", "oil_price_change", "macd"])
# plot_distributions_of_features_updated("sample_size_23_stationary_volume_fixed", ["log_return", "log_volume_change", "rsi", "vwap", "ema", "eur_usd", "oil_price", "macd"])
# compute_oil_change("sample_size_23_stationary_volume_fixed")

# compute_currency_change("sample_size_23_stationary_volume_fixed")


#plot_values_with_standard_deviation(company_name="BMW", std = 0.0051, experiment=10)
# get_amount_of_stock_changes(1.01)
# plot_training_sample_distribution("14", 22)
# model = Model11(input_size=1, hidden_size=50)
# plot_prediction_difference_distribution(model, "11", 25)

# model = Model13(input_size=1, hidden_size=50)
# plot_prediction_difference_distribution(model, "13", 23)