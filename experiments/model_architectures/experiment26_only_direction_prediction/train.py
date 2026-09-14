import os
import pandas as pd
import torch
from torch import nn, optim
from test_model import test_model
import utils.experiment_utils as utils
from Dataset import TechnicalIndicatorsDataSetFeaturesSplit
from Model import ModelLessLSTMs as Model

def train(sample_size_stocks:int, batch_size:int, hidden_size:int, learning_rate:float, momentum:float, epochs:int):
    if os.path.exists(os.path.join('outputs', 'results.csv')):
        result_df = pd.read_csv(os.path.join('outputs', 'results.csv'))
    else:
        result_df = pd.DataFrame(columns=["hidden_size", "epochs", "learning_rate", "duration", "final_train_loss", "final_test_loss", "test_direction_percentage"])

    dataset = TechnicalIndicatorsDataSetFeaturesSplit(sample_size_stocks)
    correct_direction = 10
    i = 0
    # 10 models are being created
    while i < 10:
        i += 1
        model = Model(input_size=1, hidden_size=hidden_size)
        criterion = nn.BCEWithLogitsLoss()
        # optimizer = optim.SGD([
        #     {"params": [
        #         *model.layers_lstms.parameters(),
        #         *model.dnns.parameters(),
        #         *model.ema_dnn_layer.parameters(),
        #         *model.layers_lstms_milstm.parameters(),
        #         *model.MI_LSTM_layer.parameters(),
        #         *model.DNN_Layer2.parameters(),
        #         *model.DNN_Layer3.parameters(),
        #     ], "lr": 0.04},
        #
        #     {"params": [
        #         *model.return_direction_dnn1.parameters(),
        #         *model.return_direction_dnn2.parameters(),
        #         *model.return_direction_dnn3.parameters(),
        #     ], "lr": 0.04},
        # ], lr=learning_rate, momentum=momentum)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

        final_train_loss, final_test_loss, duration, model = utils.train_model(model, dataset, batch_size, epochs, "26", criterion, optimizer)
        correct_direction_new = test_model(model, sample_size_stocks)
        print(f"------------------------------------------------\n\n {i} \n\n \n\n correct_direction_%: {correct_direction_new}")
        if correct_direction_new < correct_direction:
            torch.save(model.state_dict(), os.path.join("outputs", f"trained_model_experiment26_{round(correct_direction_new, 2)}.pth"))
            correct_direction = correct_direction_new

        result_df.loc[len(result_df)] = [hidden_size, epochs, learning_rate, duration, final_train_loss, final_test_loss, correct_direction_new]

    result_df.to_csv(os.path.join("outputs", "results.csv"), index=False)


train (
    sample_size_stocks=23,
    batch_size=16,
    hidden_size=20,
    learning_rate=0.01,
    momentum=0.9,
    epochs=3
)