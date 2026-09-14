import os
import pandas as pd
import torch
from test_model import test_model
import utils.experiment_utils as utils
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import ModelLessLSTMs as Model

def train(sample_size_stocks:int, batch_size:int, hidden_size:int, learning_rate:float, momentum:float, epochs:int):
    if os.path.exists(os.path.join('outputs', 'results.csv')):
        result_df = pd.read_csv(os.path.join('outputs', 'results.csv'))
    else:
        result_df = pd.DataFrame(columns=["hidden_size", "epochs", "learning_rate", "duration", "final_train_loss", "final_test_loss", "test_std", "test_std_percentage"])

    dataset = TechnicalIndicatorsDataSetFeaturesSplit(sample_size_stocks)
    std_in_percentage = 10
    i = 0
    # 10 models are being created
    while i < 10:
        i += 1
        model = Model(input_size=1, hidden_size=hidden_size)
        final_train_loss, final_test_loss, duration, model = utils.train_model_mtl(model, dataset, batch_size, learning_rate, momentum, epochs, "25")
        std, std_in_percentage_new = test_model(model, sample_size_stocks)
        print(f"------------------------------------------------\n\n {i} \n\n std: {std} \n\n std_%: {std_in_percentage_new}")
        if std_in_percentage_new < std_in_percentage:
            torch.save(model.state_dict(), os.path.join("outputs", f"trained_model_experiment25_{round(std_in_percentage_new, 2)}.pth"))
            std_in_percentage = std_in_percentage_new

        result_df.loc[len(result_df)] = [hidden_size, epochs, learning_rate, duration, final_train_loss, final_test_loss, std, std_in_percentage_new]

    result_df.to_csv(os.path.join("outputs", "results.csv"), index=False)


train (
    sample_size_stocks=23,
    batch_size=16,
    hidden_size=50,
    learning_rate=0.04,
    momentum=0.9,
    epochs=120
)