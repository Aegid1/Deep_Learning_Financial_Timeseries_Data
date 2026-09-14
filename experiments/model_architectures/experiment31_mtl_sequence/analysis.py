import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt


def plot_performances(single_lstm_path:str, mtl_lstm_path:str, mtl_lstm_weights_path:str):
    df_single_lstm = pd.read_csv(single_lstm_path)
    df_mtl_lstm = pd.read_csv(mtl_lstm_path)
    df_mtl_lstm_weights = pd.read_csv(mtl_lstm_weights_path)
    plt.boxplot(
        [df_single_lstm[f"final_test_loss"], df_mtl_lstm[f"final_test_loss"], df_mtl_lstm_weights[f"final_test_loss"]],
        labels=["single_lstm", "mtl_lstm", "mtl_lstm_weights"]
    )
    plt.title(f'test performances')
    plt.xlabel('models')
    plt.ylabel(f'final_test_loss')
    plt.savefig(f"test_performances.png", format='png', dpi=300)
    plt.show()


def scatterplot_features(single_lstm_path:str, mtl_lstm_path:str, mtl_lstm_weights_path:str, first_feature, second_feature):
    df_single_lstm = pd.read_csv(single_lstm_path)
    df_mtl_lstm = pd.read_csv(mtl_lstm_path)
    df_mtl_lstm_weights = pd.read_csv(mtl_lstm_weights_path)

    plt.scatter(df_single_lstm[first_feature], df_single_lstm[second_feature], label="single_lstm", c="green")
    plt.scatter(df_mtl_lstm[first_feature], df_mtl_lstm[second_feature], label="mtl_lstm", c="red")
    plt.scatter(df_mtl_lstm_weights[first_feature], df_mtl_lstm_weights[second_feature], label="mtl_lstm_task_weights", c="blue")
    plt.xlabel(first_feature)
    plt.ylabel(second_feature)
    plt.legend()
    plt.savefig(f"{first_feature}_{second_feature}.png", format='png', dpi=300)
    plt.show()


def plot_test_loss_to_train_test_data_std_diff(single_lstm_path:str, mtl_lstm_path:str, mtl_lstm_weights_path:str):
    df_single_lstm = pd.read_csv(single_lstm_path)
    df_mtl_lstm = pd.read_csv(mtl_lstm_path)
    df_mtl_lstm_weights = pd.read_csv(mtl_lstm_weights_path)
    lstm_diff = np.abs(df_single_lstm["test_data_std"] - df_single_lstm["train_data_std"])
    mtl_lstm_diff = np.abs(df_mtl_lstm["test_data_std"] - df_mtl_lstm["train_data_std"])
    mtl_lstm_weights_diff = np.abs(df_mtl_lstm_weights["test_data_std"] - df_mtl_lstm_weights["train_data_std"])

    plt.scatter(df_single_lstm["final_test_loss"], lstm_diff, label="single_lstm", c="green")
    plt.scatter(df_mtl_lstm["final_test_loss"], mtl_lstm_diff, label="mtl_lstm", c="red")
    plt.scatter(df_mtl_lstm_weights["final_test_loss"], mtl_lstm_weights_diff, label="mtl_lstm_task_weights", c="blue")
    plt.xlabel("final_test_loss")
    plt.ylabel("train_test_data_std_diff")
    plt.legend()
    plt.savefig(f"train_test_data_std_diff.png", format='png', dpi=300)
    plt.show()


# plot_test_loss_to_train_test_data_std_diff(
#     os.path.join("singleLSTM", "outputs", "results.csv"),
#     os.path.join("MultiTask", "outputs", "results.csv"),
#     os.path.join("MultiTaskWithLossWeights", "outputs", "results.csv"),
# )
#
# plot_performances(
#     os.path.join("singleLSTM", "outputs", "results.csv"),
#     os.path.join("MultiTask", "outputs", "results.csv"),
#     os.path.join("MultiTaskWithLossWeights", "outputs", "results.csv")
# )

# scatterplot_features(
#     os.path.join("singleLSTM", "outputs", "results.csv"),
#     os.path.join("MultiTask", "outputs", "results.csv"),
#     os.path.join("MultiTaskWithLossWeights", "outputs", "results.csv"),
#     "final_test_loss",
#     "test_data_std"
# )

# scatterplot_features(
#     os.path.join("singleLSTM", "outputs", "results.csv"),
#     os.path.join("MultiTask", "outputs", "results.csv"),
#     os.path.join("MultiTaskWithLossWeights", "outputs", "results.csv"),
#     "final_test_loss",
#     "train_data_std"
# )