import math
import numpy as np
import seaborn
import pandas as pd
import os
import matplotlib.pyplot as plt


def visualize_distribution_of_ensemble_predictions(input_path:str, output_path:str, model_names:list[str], pred_type:str):
    for model in model_names:
        model_path = os.path.join(input_path, model)
        files = [f for f in os.listdir(model_path) if f.endswith('.csv')]

        n_files = len(files)
        if n_files == 0:
            print(f"no files found in: {model_path}")
            continue

        # determine grid size of final plot
        cols = 3
        rows = math.ceil(n_files / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten() # flatten to index later with single index

        for idx, file in enumerate(files):
            # header=None, because the files don't contain a row with the columns names
            file_path = os.path.join(input_path, model, file)
            df = pd.read_csv(file_path)
            ax = axes[idx]

            if pred_type == 'mean':
                # seaborn.kdeplot(data=df['mean'], fill=True, label='Means', ax=ax)
                seaborn.histplot(df['mean'], color="blue", label="means", kde=True, stat="density", bins=50, ax=ax)
            elif pred_type == 'std':
                # seaborn.kdeplot(data=df['std'], fill=True, label='Stds', ax=ax)
                seaborn.histplot(df['std'], color="blue", label="std", kde=True, stat="density", bins=50, ax=ax)
            else:
                print("provided plot_type is not available: ", pred_type)


            ax.set_title(f'{file.replace(".csv", "")}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

        # delete all remaining parts of the grid that arent used
        for i in range(n_files, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f'Distributions of all ensemble predictions - Model: {model}', fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_path, f'{model}_ensemble_preds_{pred_type}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()


def visualize_distribution_of_actual_values(input_path:str, output_path:str):
    files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    n_files = len(files)

    if n_files == 0:
        print(f"no files found in: {input_path}")
        return

    cols = 3
    rows = math.ceil(n_files / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()  # flatten to index later with single index

    for idx, file in enumerate(files):
        # header=None, because the files don't contain a row with the columns names
        file_path = os.path.join(input_path, file)
        df = pd.read_csv(file_path)
        ax = axes[idx]

        # seaborn.kdeplot(data=np.exp(df['log_return']), fill=True, label='Means', ax=ax)
        seaborn.histplot(np.exp(df['log_return']), color="blue", label="means", kde=True, stat="density", bins=50, ax=ax)

        ax.set_title(f'{file.replace(".csv", "")}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # delete all remaining parts of the grid that arent used
    for i in range(n_files, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f'Distributions of all actual returns', fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_path, f'actual_returns.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# visualize_distribution_of_ensemble_predictions(
#     input_path='outputs',
#     output_path=os.path.join("outputs", "analysis"),
#     model_names=["MILSTM_sample_reweighting_less_weight_decay"],
#     pred_type="mean"
# )


# visualize_distribution_of_actual_values(
#     input_path=os.path.join("..", "..", "..", "data", "stocks_test_data_benchmark"),
#     output_path=os.path.join("outputs", "analysis")
# )