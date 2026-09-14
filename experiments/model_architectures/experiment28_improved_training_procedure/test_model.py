import math
import os
import pickle
import pandas as pd
import torch
from matplotlib import pyplot as plt
import numpy as np
from Model import ModelLessLSTMs as Model
from Dataset import scale_inputs_and_labels

def test_model_log_returns(model, sample_size: int):
    model.eval()
    pickle_directory = os.path.join("..", "..", "data", "samples", "test_samples", f"sample_size_{sample_size}_stationary_volume_fixed")
    company_names = os.listdir(pickle_directory)
    differences_in_percentage, differences = [], []
    predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
    for company in company_names:
        pkl_files = [f for f in os.listdir(os.path.join(pickle_directory, company)) if f.endswith('.pkl')]
        correct_direction_count = 0
        for file in pkl_files:
            file_path = os.path.join(pickle_directory, company, file)
            with open(file_path, 'rb') as f:
                item = pickle.load(f)
            features_dict, target_tensor = scale_inputs_and_labels(item)

            log_prediction = model.forward(**features_dict)
            prediction = math.exp(log_prediction)

            actual_value = math.exp(item['target'])
            prediction_date = item['datetime']
            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [prediction],
                "Actual": [actual_value]
            })
            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

            previous_value = math.exp(item["features"]["stock_prices"][-1])
            if is_prediction_in_correct_direction(prediction, actual_value, previous_value):
                correct_direction_count += 1

        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        predictions_df = predictions_df.sort_values(by="Date").reset_index(drop=True)
        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference_percentage = predictions_df['Difference'].mean() * 100

        print(f"{company} in %: {average_difference_percentage:.3f}%")
        print(f"{company}: correct direction of prediction in %: {correct_direction_count / len(predictions_df['Actual']):.4f}")
        differences_in_percentage.append(average_difference_percentage)

    overall_difference_percentage = sum(differences_in_percentage)/len(differences_in_percentage)
    return overall_difference_percentage

def create_single_prediction_plot(model_path: str, sample_size: int, company_name:str):
    model = Model(input_size=1, hidden_size=50)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    pickle_directory = os.path.join("..", "..", "data", "samples", "test_samples", f"sample_size_{sample_size}_stationary_volume_fixed", company_name)
    pkl_files = [f for f in os.listdir(pickle_directory) if f.endswith('.pkl')]

    predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
    for file in pkl_files:
        file_path = os.path.join(pickle_directory, file)
        with open(file_path, 'rb') as f:
            item = pickle.load(f)
        features_dict, target_tensor, stock_scaler = scale_inputs_and_labels(item)

        prediction = model.forward(**features_dict)
        prediction_denormalized = stock_scaler.inverse_transform(prediction.detach().numpy())[0][0]

        actual_value = item['target']
        prediction_date = item['datetime']
        new_row = pd.DataFrame({
            "Date": [prediction_date],
            "Prediction": [prediction_denormalized],
            "Actual": [actual_value]
        })
        predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

    predictions_df = predictions_df.sort_values(by="Date")
    predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
    average_difference = predictions_df['Difference'].mean()
    average_price = predictions_df['Actual'].mean()
    average_difference_percentage = (average_difference / average_price) * 100
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
    plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Model Predictions vs Actual Values")
    plt.legend(title=f"Average Difference in %: {average_difference_percentage:.4f}")
    save_path = os.path.join(f"{"/".join(model_path.split("/")[:-1])}", "test_model", f"moving_average_prediction{company_name}.png")
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()


def test_prediction(model_path: str, sample_size: int):
    overall_average_difference = 0
    model = Model(input_size=1, hidden_size=50)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    pickle_directory = os.path.join("..", "..", "data", "samples", "test_samples", f"sample_size_{sample_size}_stationary_volume_fixed")
    company_names = os.listdir(pickle_directory)

    for company in company_names:
        pickle_directory_with_company = os.path.join(pickle_directory, company)
        pkl_files = [f for f in os.listdir(pickle_directory_with_company) if f.endswith('.pkl')]
        predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
        correct_direction_count = 0
        for file in pkl_files:
            file_path = os.path.join(pickle_directory_with_company, file)
            with open(file_path, 'rb') as f:
                item = pickle.load(f)

            features_dict, target_tensor, stock_scaler = scale_inputs_and_labels(item)

            # denormalize the prediction
            prediction = model.forward(**features_dict)
            prediction_denormalized = stock_scaler.inverse_transform(prediction.detach().numpy())[0][0]

            actual_value = item['target']
            prediction_date = item['datetime']
            new_row = pd.DataFrame({
                "Date": [prediction_date],
                "Prediction": [prediction_denormalized],
                "Actual": [actual_value]
            })

            previous_value = item["features"]["stock_prices"][-1]
            if is_prediction_in_correct_direction(prediction_denormalized, actual_value, previous_value):
                correct_direction_count += 1

            predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)

        predictions_df = predictions_df.sort_values(by="Date")
        predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
        average_difference = predictions_df['Difference'].mean()
        average_price = predictions_df['Actual'].mean()
        average_difference_percentage = (average_difference / average_price) * 100
        overall_average_difference += average_difference_percentage
        print(f"Average difference in % for {company}: {average_difference_percentage} \n")
        print(f"correct direction of prediction in %: {correct_direction_count/len(predictions_df['Actual']):.4f}")
        plt.figure(figsize=(10, 6))
        plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
        plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Model Predictions vs Actual Values")
        plt.legend(title=f"Average Difference in %: {average_difference_percentage:.4f}")
        plt.legend(title=f"correct direction of prediction in %: {correct_direction_count/len(predictions_df['Actual']):.4f}")
        save_path = os.path.join(f"{"/".join(model_path.split("/")[:-1])}", "test_model", f"model_prediction_plot_{company}_{sample_size}.png")
        plt.savefig(save_path, format='png', dpi=300)
    print(f"overall average difference in % over all companies: {overall_average_difference / len(company_names)} \n")


def is_prediction_in_correct_direction(prediction, actual, previous):
    predicted_change = prediction - previous
    actual_change = actual - previous
    if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0):
        return True
    return False


#test_prediction(os.path.join("outputs", "trained_model_experiment20_0.53.pth"), 23)
#create_single_prediction_plot(os.path.join("outputs", "trained_model_experiment20_0.51.pth"), 23, "Airbus")