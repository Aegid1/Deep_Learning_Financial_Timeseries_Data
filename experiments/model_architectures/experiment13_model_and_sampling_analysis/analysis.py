import math
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from test_model import test_model
from Model import ModelLessLSTMs as Model

def create_time_distribution_dataframe(experiment:str, sample_size:int):
    time_distribution_df = pd.DataFrame(columns=["Company", "Day_Name", "Time"])

    path = os.path.join('..', "..", "data", "samples", "train_samples", f"sample_size_{sample_size}")
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

    time_distribution_df.to_csv(os.path.join("outputs", f"model_{experiment}","time_distribution.csv"))
    return time_distribution_df


def plot_training_sample_distribution(experiment:str, sample_size:int):
    if os.path.exists(os.path.join("outputs", f"model_{experiment}","time_distribution.csv")):
        df = pd.read_csv(os.path.join("outputs", f"model_{experiment}","time_distribution.csv"))
    else:
        df = create_time_distribution_dataframe(experiment, sample_size)

    # creates the barplot for the day distribution
    day_counts_df = df["Day_Name"].value_counts().reset_index()
    total = day_counts_df["count"].sum()
    day_counts_df["relative"] = ((day_counts_df["count"] / total) * 100).round(3)
    plt.bar(day_counts_df["Day_Name"], day_counts_df["count"])

    # adding relative values on top of bars
    for i in range(len(day_counts_df)):
        plt.text(i, day_counts_df["count"].iloc[i], f"{day_counts_df["relative"].iloc[i]} %", ha="center", fontweight="bold")  # Placing text slightly above the bar

    plt.title('Days Distribution')
    plt.xlabel('days')
    plt.ylabel('count')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "days_distribution.png"), format='png', dpi=300)
    plt.show()

    # creates the barplot for the time distribution
    time_counts_df = df["Time"].value_counts().reset_index()
    for index, row in time_counts_df.iterrows():
        print(row['Time'].split("+")[0])
        time_counts_df.loc[index, "Time"] = row['Time'].split("+")[0]
    plt.bar(time_counts_df["Time"], time_counts_df["count"])

    # adding relative values on top of bars
    time_counts_df["relative"] = ((time_counts_df["count"] / total) * 100).round(3)
    for i in range(len(time_counts_df)):
        plt.text(i, time_counts_df["count"].iloc[i], f"{time_counts_df["relative"].iloc[i]} %", ha="center", fontweight="bold")  # Placing text slightly above the bar

    plt.title('Time Distribution')
    plt.xlabel('times')
    plt.ylabel('count')
    plt.savefig(os.path.join("outputs", f"model_{experiment}", "time_distribution.png"), format='png', dpi=300)
    plt.show()


def create_prediction_difference_distribution(model, experiment:str, sample_size: int, model_path:str):
    df = test_model(model, sample_size, model_path)
    df.to_csv(os.path.join("outputs", f"model_{experiment}", "prediction_errors_distribution.csv"))
    return df


def plot_prediction_difference_distribution(model, experiment:str, sample_size: int, model_path:str):
    if os.path.exists(os.path.join("outputs", f"model_{experiment}", "prediction_errors_distribution.csv")):
        df = pd.read_csv(os.path.join("outputs", f"model_{experiment}","prediction_errors_distribution.csv"))
    else:
        df = create_prediction_difference_distribution(model, experiment, sample_size, model_path)

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


def get_amount_of_stock_changes(stock_change_in_percentage:float):
    values = []
    companies = [
        "Airbus",
        "Allianz",
        "BMW",
        "Deutsche Bank",
        "Deutsche Telekom",
        "Mercedes-Benz",
        "Porsche",
        "SAP",
        "Siemens",
        "Siemens Healthineers",
        "Volkswagen"
    ]

    average_working_days = 252
    average_timesteps_in_a_year = average_working_days * 5  # 5 timesteps in a day

    for company in companies:
        df = pd.read_csv(os.path.join("..", "..", "data", "stocks", f"{company}.csv"))
        changes = df["Open"] / df["Open"].shift(1)
        changes = changes.round(4)
        changes_greater_one_percent = changes[(changes > stock_change_in_percentage)]
        total_amount_of_timesteps = len(changes)
        amount_of_years = math.ceil(total_amount_of_timesteps / average_timesteps_in_a_year)
        average_amount_of_changes_greater_one_percent = len(changes_greater_one_percent) / amount_of_years
        values.append(average_amount_of_changes_greater_one_percent)
        print(average_amount_of_changes_greater_one_percent)

    print(sum(values)/len(values))


# get_amount_of_stock_changes(1.01)
# plot_training_sample_distribution("14", 22)
# model = Model(input_size=1, hidden_size=50)
# model = Model(input_size=1, hidden_size=50)

# model with bias
# model = Model(input_size=1, hidden_size=50)
# plot_prediction_difference_distribution(model, "13", 25, os.path.join("sampling_bias", "outputs", ""))


# model without bias
# model = Model(input_size=1, hidden_size=50)
# plot_prediction_difference_distribution(model, "13", 23, os.path.join("no_sampling_bias", "outputs", ""))