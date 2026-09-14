import os
import pickle
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def scale_inputs_and_labels(item):
    features_lstms, features = [], []
    relevant_features = ["stock_prices", "rsi", "macd", "vwap", "ema"]

    for feature_name, feature_value in item['features'].items():
        scaler = MinMaxScaler()
        if feature_name in relevant_features:
            feature_normalized = scaler.fit_transform(np.array(feature_value).reshape(-1, 1))
            feature_tensor = torch.tensor(feature_normalized, dtype=torch.float32)
            if feature_name == "ema":
                features.append(feature_tensor)
            else:
                features_lstms.append(torch.tensor(feature_tensor, dtype=torch.float32))

    features = {
        "features_lstm": torch.tensor(np.array(features_lstms), dtype=torch.float32),
        "features": torch.tensor(np.array(features), dtype=torch.float32)
    }
    target = np.array(item['target'])
    target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1)  # [batch_size, 1]
    return features, target_tensor, {}


def test_model(model, sample_size: int, model_path:str):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    path = os.path.join("..", "..", "..", "data", "samples", "leftover_data", "test_samples")
    company_names = os.listdir(path)

    predictions_df = pd.DataFrame(columns=["Day", "Time", "Prediction", "Predicted_Change_Direction", "Actual_Change_Direction", "Actual", "Difference"])
    for company in company_names:
        print(company)
        print("---------------------")
        pickle_directory = os.path.join(path, company)
        pkl_files = [f for f in os.listdir(pickle_directory) if f.endswith('.pkl')]

        for file in pkl_files:
            file_path = os.path.join(pickle_directory, file)
            with open(file_path, 'rb') as f:
                item = pickle.load(f)

            inputs, target_tensor, _ = scale_inputs_and_labels(item)
            prediction_log = model.forward(**inputs) # predicts the log_price
            prediction = torch.exp(prediction_log)

            actual_value = item['target']
            day_name = pd.to_datetime(item['datetime']).day_name()
            time = str(item['datetime']).split(" ")[1]
            time_without_timezone = time.split("+")[0]
            last_stock_price = item['features']["stock_prices"][-1]
            predicted_change_direction = prediction - last_stock_price
            actual_change_direction = actual_value - last_stock_price

            new_row = {
                "Company": company,
                "Day": day_name,
                "Time": time_without_timezone,
                "Prediction": prediction,
                "Predicted_Change_Direction": predicted_change_direction,
                "Actual_Change_Direction": actual_change_direction,
                "Actual": actual_value,
                "Difference": prediction - actual_value,
            }
            predictions_df.loc[len(predictions_df)] = new_row

    return predictions_df
