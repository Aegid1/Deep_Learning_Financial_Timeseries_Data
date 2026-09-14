import os
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import utils.experiment_utils as utils
from Dataset import Dataset
from Model import ModelLessLSTMs as Model

def train(
        sample_size_stocks:int,
        maximal_batch_size:int,
        hidden_size:int,
        initial_learning_rate:float,
        epochs:int,
        experiment_number: int,
        samples_name:str,
        results_path:str
):
    result_df_path = os.path.join(results_path, "results.csv")
    dataset = Dataset(sample_size_stocks, samples_name)
    for i in range(3):
        result_df = utils.create_results_csv(
            columns=["hidden_size", "params", "initial_learning_rate", "duration", "final_train_loss", "final_test_loss"],
            path= result_df_path
        )
        model = Model(input_size=1, hidden_size=hidden_size)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) # calculates total amount of params
        optimizer = torch.optim.AdamW(model.parameters(), initial_learning_rate, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-4)
        train_loader, test_loader = utils.create_train_test_split(dataset, maximal_batch_size)

        final_train_loss, final_test_loss, _, duration, model = utils.train_model(
            model, epochs, nn.MSELoss(), optimizer, train_loader, test_loader, scheduler
        )
        torch.save(model.state_dict(), os.path.join(results_path, f"trained_model_experiment{experiment_number}_{final_test_loss}.pth"))

        # add the result to the existing ones and save the updated dataframe as a .csv
        result_df.loc[len(result_df)] = [
            hidden_size, params, initial_learning_rate, duration, final_train_loss, final_test_loss
        ]
        result_df.to_csv(result_df_path, index=False)



# train (
#     sample_size_stocks=23,
#     maximal_batch_size=32,
#     hidden_size=50,
#     initial_learning_rate=0.005,
#     epochs=100,
#     experiment_number=13,
#     samples_name="sample_size_23_news_combined"
# )