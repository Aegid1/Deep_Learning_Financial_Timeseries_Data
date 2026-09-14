import os
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import Dataset
from _optuna_optimization.ModelObjective import ModelObjective
from _optuna_optimization.stationary_volume_new.Dataset import TechnicalIndicatorsDataSetFeaturesSplit
import optuna_model_factory

MODEL_CLASSES = {
    "ALSTM": optuna_model_factory.get_alstm_builder,
    "GAT": optuna_model_factory.get_gat_builder,
    "GRU": optuna_model_factory.get_gru_builder,
    "Localformer": optuna_model_factory.get_localformer_builder,
    "TCN": optuna_model_factory.get_tcn_builder,
    "Transformer": optuna_model_factory.get_transformer_builder,
}

def perform_optuna_study(dataset:Dataset, model_name:str):
    model_builder_func = MODEL_CLASSES[model_name]

    objective = ModelObjective(
        model_builder_func=model_builder_func,
        dataset=dataset,
        maximal_batch_size=32,
        epochs=100,
        runs_per_trial=4
    )

    study = optuna.create_study(direction="minimize", study_name=f"hyperparam_search_{model_name}")
    study.optimize(objective, n_trials=5) # n_trials is amount of different parameter configurations that are being tested

    print("best parameters:", study.best_params)
    print("smallest average deviation:", study.best_value)
    return study


def plot_optuna_results(study, model_name, feature_set_name):
    # extract the data of the optuna study
    data = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        config_label = trial.user_attrs["config_label"]
        deviations = trial.user_attrs["all_deviations"]

        for dev in deviations:
            data.append({
                "Configuration": config_label,
                "Final Deviation": dev
            })

    df = pd.DataFrame(data)
    plt.figure(figsize=(14, 7))

    # plots the result points for every configuration
    sns.stripplot(
        data=df,
        x="Configuration",
        y="Final Deviation",
        size=6,
        alpha=0.7,
        hue="Configuration",
        legend=False
    )

    plt.title(f"Parameter-Configurations vs. Final Deviation ({model_name})", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(feature_set_name, f"{model_name}.png"))
    plt.show()


optuna_study = perform_optuna_study(
    dataset=TechnicalIndicatorsDataSetFeaturesSplit(
        23,
        os.path.join("..", '..', '..', "..", 'data', 'samples', 'train_samples', "stationary_volume_new")
    ),
    model_name="Transformer"
)
plot_optuna_results(study=optuna_study, model_name="Transformer", feature_set_name="stationary_volume_new")