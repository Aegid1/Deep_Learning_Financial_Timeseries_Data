import os
import pickle
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from Model import ModelLessLSTMs as Model

def scale_inputs_and_labels(item):
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
        "features_lstm": torch.tensor(np.array(features_lstms), dtype=torch.float32).unsqueeze(0),
        "features": torch.tensor(np.array(features), dtype=torch.float32).unsqueeze(0)
    }
    target = np.array(item['target'])
    target_tensor = torch.tensor(np.log(target), dtype=torch.float32).unsqueeze(-1).unsqueeze(0)  # [batch_size, 1]
    return features, target_tensor, {}


def test_model(model_parameters_path:str, test_samples_path:str, model):
    state_dict = torch.load(model_parameters_path)
    model.load_state_dict(state_dict)
    model.eval()
    pkl_files = [f for f in os.listdir(test_samples_path) if f.endswith('.pkl')]

    predictions_df = pd.DataFrame(columns=["Date", "Prediction", "Actual"])
    for file in pkl_files:
        file_path = os.path.join(test_samples_path, file)
        with open(file_path, 'rb') as f:
            item = pickle.load(f)
            # as these are not processed in batches, an extra dimension was added in scale_inputs_and_labels()
            inputs, target, _ = scale_inputs_and_labels(item)
            prediction_log = model.forward(**inputs)
            prediction = torch.exp(prediction_log)

        actual_value = item['target']
        prediction_date = item['datetime']
        new_row = pd.DataFrame({
            "Date": [prediction_date],
            "Prediction": [prediction.item()],
            "Actual": [actual_value]
        })
        predictions_df = pd.concat([predictions_df, new_row], ignore_index=True)


    predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
    predictions_df = predictions_df.sort_values(by="Date").reset_index(drop=True)
    predictions_df['Difference'] = abs(predictions_df['Prediction'] - predictions_df['Actual'])
    average_difference = predictions_df['Difference'].mean()
    print(predictions_df["Difference"])
    print(average_difference)
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_df['Date'], predictions_df['Prediction'], label='Prediction', color='blue')
    plt.plot(predictions_df['Date'], predictions_df['Actual'], label='Actual', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Model Predictions vs Actual Values")
    plt.legend(title=f"Average Difference: {average_difference:.4f}")
    save_path = os.path.join("outputs", "prediction_plot_experiment10.png")
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()


test_model(
    model_parameters_path=os.path.join("outputs", "trained_model_experiment10_0.7565068304538727.pth"),
    test_samples_path=os.path.join("..", "..", "..", "data", "samples", "leftover_data", "test_samples"),
    model=Model(input_size=1, hidden_size=50)
)
