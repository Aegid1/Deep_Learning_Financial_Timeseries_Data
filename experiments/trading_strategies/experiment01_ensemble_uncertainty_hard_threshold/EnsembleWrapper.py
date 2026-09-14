import numpy as np
import torch
from benchmark_models.ALSTM.ALSTM import ALSTMModel
from benchmark_models.GAT.GAT import GATM
from benchmark_models.GRU.GRU import GRUModel
from benchmark_models.Localformer.Localformer import Transformer as Localformer
from benchmark_models.TCN.TCN import TCNModel
from benchmark_models.Transformer.Transformer import Transformer
from experiments.model_architectures.experiment30_smooth_l1_loss.Model import ModelLessLSTMs as MILSTM

class EnsembleWrapper:
    def __init__(self, model_paths: list[str], model_class:str, top_k_models:int=5):
        """
        Loads multiple models for an ensemble
        :param model_paths: list of paths to the model weights (.pt / .pth)
        :param model_class: the class of the model
        """
        self.models = []
        class_dict = {
            "ALSTM": lambda: ALSTMModel(d_feat=7, hidden_size=32, num_layers=4, dropout=0.0),
            "GAT": lambda: GATM(d_feat=7, hidden_size=128, num_layers=4, dropout=0.0),
            "GRU": lambda: GRUModel(d_feat=7, hidden_size=128, num_layers=6, dropout=0.0),
            "Localformer": lambda: Localformer(d_feat=7, d_model=32, nhead=8, num_layers=6, dropout=0.0),
            "TCN": lambda: TCNModel,
            "Transformer": lambda: Transformer(d_feat=7, d_model=64, nhead=2, num_layers=3, dropout=0.0),
            "MILSTM": lambda: MILSTM(hidden_size=40)
        }

        # paths are sorted, because their name show the error of the model -> models are sorted ascending
        # thus best models are the first of the sorted list
        paths = sorted(model_paths)
        print(paths)
        for i in range(min(top_k_models, len(paths))): # min handles the case for having less than top_k_models
            # call the lambda function to create the model and load the params
            model = class_dict[model_class]()
            print(paths[i])
            model.load_state_dict(torch.load(paths[i]))
            model.eval()
            self.models.append(model)


    def predict(self, inputs:dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs a forward pass of all models.
        :return: mean and std of the predictions
        """
        predictions = []

        with torch.no_grad():
            for model in self.models:
                # ASSUMPTION: all considered model need to handle a dict with the features in the forward() func
                out = model(**inputs)
                predictions.append(out.numpy())

        predictions_stacked_log = np.array(predictions) # Stacked Array, Shape: (num_models, batch_size, output_dim)
        # the models are currently trained to predict the log return -> thus np.exp() is necessary
        predictions_stacked = np.exp(predictions_stacked_log)

        mean_preds = np.mean(predictions_stacked, axis=0).squeeze()
        std_preds = np.std(predictions_stacked, axis=0).squeeze()
        return mean_preds, std_preds