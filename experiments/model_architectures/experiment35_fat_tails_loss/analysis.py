import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
import torch
from scipy.stats import rankdata
from tercile_weights.Model import ModelLessLSTMs as Model

# assumes that the data are the log returns
def exponential_weights(data, alpha=10.0):
    return np.exp(alpha * np.abs(data))

# assumes the median of the returns is around 0 and the data are the log returns
def tercile_weights(data, w=2.0, tau=0.05):
    #TODO add doc describing the general idea of using a sigmoid function and the roles of parameters
    u = rankdata(data) / len(data)                   # ranks the values and maps them relative to interval (0,1)
    d = np.abs(u - 0.5)                              # distance for each point to the median-center
    w = 1 + (w - 1) / (1 + np.exp(-(d - 1/6) / tau)) # weights based on a sigmoid function
    return w / w.sum()


def calculate_weights(data, weighting_method: str):
    if weighting_method == "exponential":
        return exponential_weights(data)
    elif weighting_method == "tercile":
        return tercile_weights(data)
    else:
        print(f"weighting method: {weighting_method} not implemented")
        return None


def visualize_weighted_distributions(data_path:str, output_path:str, weighting_method:str):
    files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    n_files = len(files)

    if n_files == 0:
        print(f"no files found in: {data_path}")
        return

    cols = 2 # two cols to have side by side comparisons of actual and weighted distributions
    rows = n_files
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # flatten to index later with single index

    for i, file in enumerate(files):
        # header=None, because the files don't contain a row with the columns names
        file_path = os.path.join(data_path, file)
        df_real_data = pd.read_csv(file_path)

        # plotting actual data
        ax = axes[i*2]
        seaborn.histplot(df_real_data['log_return'], color="blue", label="means", kde=True, stat="density", bins=50, ax=ax)

        ax.set_title(f'{file.replace(".csv", "")}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

        # plotting the transformed data with sample weights
        ax = axes[i*2+1]
        data = df_real_data['log_return']
        weights = calculate_weights(data, weighting_method)
        seaborn.histplot(x=data, weights=weights, color="blue", kde=True, stat="density", bins=50, ax=ax)

        ax.set_title(f'{file.replace(".csv", "")} weighted')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(f'Distributions of actual and weighted distributions', fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_path, f'distributions_difference_{weighting_method}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def visualize_learned_weight_distributions_of_model(model, weights_path, include_biases=True, bins=100):
    model.load_state_dict(torch.load(weights_path))
    all_weights = []

    for name, param in model.named_parameters():
        if not include_biases and 'weight' not in name:
            continue

        weights = param.detach().cpu().numpy().flatten()
        all_weights.append(weights)

    all_weights_concat = np.concatenate(all_weights)

    plt.figure(figsize=(10, 6))
    plt.hist(all_weights_concat, bins=bins, color='blue', alpha=0.7, density=True)

    plt.title("Distribution of learned model weights", fontsize=14)
    plt.xlabel("Weight value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    plt.show()


# visualize_weighted_distributions(
#     data_path=os.path.join("..", "..", "..", "data", "stocks_test_data_benchmark"),
#     output_path=os.path.join("exponential_weights", "outputs"),
#     weighting_method="exponential"
# )

# visualize_weighted_distributions(
#     data_path=os.path.join("..", "..", "..", "data", "stocks_test_data_benchmark"),
#     output_path=os.path.join("tercile_weights", "outputs"),
#     weighting_method="tercile"
# )

# visualize_learned_weight_distributions_of_model(
#     model=Model(hidden_size=40),
#     weights_path=os.path.join("tercile_weights", "outputs", 'trained_model_experiment35_0.588.pth'),
#     include_biases=False,
# )