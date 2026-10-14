import datetime
import json
import math
import os
import time
import pandas as pd
import torch
from matplotlib.lines import Line2D
from torch import optim, nn
from torch.utils.data import random_split, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import plotext as plt_console

#------------------------------------------STANDARD-MODEL-TRAINING------------------------------------------
def create_train_test_split(dataset, batch_size:int):
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def perform_single_train_iteration(model_inputs, labels, optimizer: optim.Optimizer, model, running_training_loss:float, criterion, max_norm: float = 5.0):
    optimizer.zero_grad()
    outputs = model.forward(**model_inputs) if isinstance(model_inputs, dict) else model.forward(model_inputs)

    loss = model.calculate_loss(criterion, outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    gradient_norm = check_gradients(model)

    optimizer.step()
    running_training_loss += loss.item()
    return running_training_loss, gradient_norm


def perform_single_test_iteration(model_inputs, labels, model, running_testing_loss:float, criterion):
    outputs = model.forward(**model_inputs) if isinstance(model_inputs, dict) else model.forward(model_inputs)

    loss = model.calculate_loss(criterion, outputs, labels)
    running_testing_loss += loss.item()
    # ASSUMPTION: model predicts log-change or log-price
    diffs = np.abs(np.exp(outputs.squeeze(-1)) - np.exp(labels.squeeze(-1)))
    return running_testing_loss, diffs


def train_model(
        model,
        epochs:int,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        lr_scheduler=None,
        enable_logs: bool = True
):
    start_time = time.time()
    train_loss_vals, test_loss_vals = list(), list()
    epoch_percentage_deviation = 0

    for epoch in range(epochs):
        running_training_loss, running_test_loss = 0, 0
        diffs_array = np.array([])

        model.train()
        gradient_norms = []
        for inputs, labels, _ in train_loader:
            running_training_loss, gradient_norm = perform_single_train_iteration(inputs, labels, optimizer, model, running_training_loss, criterion)
            gradient_norms.append(gradient_norm)
        train_loss_vals.append(running_training_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            #TODO sind das wirklich attention aktivierungen im klassischen sinne?
            if (epoch == epochs-1) and hasattr(model, "MI_LSTM_layer"): model.MI_LSTM_layer.add_global_attentions = True  # track attentions in last epoch

            for inputs, labels, _ in test_loader:
                running_test_loss, diffs = perform_single_test_iteration(inputs, labels, model, running_test_loss, criterion)
                diffs_array = np.concatenate((diffs_array, diffs))
            test_loss_vals.append(running_test_loss / len(test_loader))

        if math.isnan(running_training_loss): # something in the training failed
            print(f"Some numerical instability occurred: {running_training_loss}")
            train_loss_vals, test_loss_vals = [0] * epochs, [0] * epochs
            break

        if lr_scheduler: lr_scheduler.step() # when lr_scheduler is used apply step here

        epoch_percentage_deviation = round((np.mean(diffs_array)) * 100, 3)
        if enable_logs:
            print(f"EPOCH: {epoch}")
            print(f"Training loss: {running_training_loss / len(train_loader)}")
            print(f"Test loss: {running_test_loss / len(test_loader)}")
            print(f"Percentual deviation of stock_price: {epoch_percentage_deviation}%")
            print(f"Average gradient norm: {sum(gradient_norms)/len(gradient_norms):.6f}")
            print(f"Current learning rate: {lr_scheduler.get_last_lr()[0]}")

    duration = time.time() - start_time
    print(f"training took {duration:.2f} seconds.")
    # plot_train_test_loss_values(train_loss_vals, test_loss_vals, epochs)
    return train_loss_vals[-1], test_loss_vals[-1], epoch_percentage_deviation, duration, model
#-------------------------------------------------------------------------------------------------------------


#------------------------------------------MULTI-TASK-MODEL-TRAINING------------------------------------------
def perform_single_train_iteration_mtl(model_inputs:dict, labels, optimizer: optim.Optimizer, model, running_training_loss:float):
    optimizer.zero_grad()
    outputs = model.forward(**model_inputs)
    loss = model.calculate_loss(outputs, labels)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    gradient_norm = check_gradients(model)

    optimizer.step()
    running_training_loss += loss.item()
    return { "running_train_loss": running_training_loss, "gradient_norm": gradient_norm }


def perform_single_test_iteration_mtl(model_inputs:dict, labels, model, running_testing_loss:float):
    outputs = model.forward(**model_inputs) # outputs.shape = [batch=32, outputs=4]
    loss = model.calculate_loss(outputs, labels)
    running_testing_loss += loss.item()
    diffs = np.abs(outputs[:,0].squeeze(-1) - labels[:,0].squeeze(-1)) # first output of a batch is for return
    return { "running_test_loss": running_testing_loss, "diffs": diffs }


# TODO check if this function is even necessary -> only the criterion is not used here the rest is the same as above
def train_model_mtl(model, epochs:int, optimizer, train_loader, test_loader, lr_scheduler=None,
                    train_step_fn=perform_single_train_iteration_mtl, test_step_fn=perform_single_test_iteration_mtl,
                    callbacks=None):

    callbacks = callbacks or []  # if None, then empty list
    start_time = time.time()
    train_loss_vals, test_loss_vals = list(), list()
    epoch_percentage_deviation = 0

    for epoch in range(epochs):
        running_training_loss, running_test_loss = 0.0, 0.0
        diffs_array = np.array([])

        model.train()
        gradient_norms = []
        for inputs, labels, _ in train_loader:
            # running_training_loss, gradient_norm = perform_single_train_iteration_mtl(inputs, labels, optimizer, model, running_training_loss)
            train_step_results = train_step_fn(inputs, labels, optimizer, model, running_training_loss)
            running_training_loss = train_step_results["running_train_loss"]
            gradient_norms.append(train_step_results["gradient_norm"])

            for cb in callbacks:
                if hasattr(cb, "on_train_batch_end"): cb.on_train_batch_end(train_step_results, epoch)

        model.print_task_weights()
        model.eval()
        with torch.no_grad():
            # TODO put that into the on_epoch_end callback of the corresponding experiment
            if (epoch == epochs-1) and hasattr(model, "mi_lstm_layer"): model.MI_LSTM_layer.add_global_attentions = True  # track attentions in last epoch

            for inputs, labels, _ in test_loader:
                # running_test_loss, diffs = test_step_fn(inputs, labels, model, running_training_loss)
                test_step_results = test_step_fn(inputs, labels, model, running_test_loss)
                running_test_loss = test_step_results["running_test_loss"]
                diffs_array = np.concatenate((diffs_array, test_step_results["diffs"]))

                for cb in callbacks:
                    if hasattr(cb, "on_test_batch_end"): cb.on_test_batch_end(test_step_results, epoch)

        test_loss_vals.append(running_test_loss / len(test_loader))
        train_loss_vals.append(running_training_loss / len(train_loader))
        epoch_percentage_deviation = np.mean(diffs_array) * 100

        torch.set_grad_enabled(True)

        if lr_scheduler: lr_scheduler.step() # when lr_scheduler is given apply step here

        print(f"EPOCH: {epoch}")
        print(f"Training loss: {running_training_loss / len(train_loader)}")
        print(f"Test loss: {running_test_loss / len(test_loader)}")
        print(f"Percentual deviation of stock_price: {epoch_percentage_deviation:.3f}%")
        print(f"Average gradient norm: {sum(gradient_norms)/len(gradient_norms):.6f}")
        print(f"Current learning rate: {lr_scheduler.get_last_lr()[0]}")


    end_time = time.time()
    duration = end_time - start_time
    print(f"The training took {duration:.2f} seconds")
    # plot_train_test_loss_values(train_loss_vals, test_loss_vals, epochs)
    return train_loss_vals[-1], test_loss_vals[-1], epoch_percentage_deviation, duration, model
#-------------------------------------------------------------------------------------------------------------


def check_gradients(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def debug_activations(x, name="tensor"):
    print(f"{name}: mean={x.mean().item():.6f}, std={x.std().item():.6f}, "
          f"min={x.min().item():.6f}, max={x.max().item():.6f}")


def create_results_csv(columns:list[str], path:str):
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return pd.DataFrame(columns=columns)


def compute_dataloader_mean_std_movement_direction(dataloader):
    total_sum, total_sum_sq = 0.0, 0.0
    total_count, total_count_labels = 0, 0
    positive_movement_directions, positive_movement_directions_target = 0, 0

    for features, labels, _ in dataloader:
        labels = labels[:,0]
        f_data = torch.exp(features["log_return"])
        total_count += f_data.numel()
        total_count_labels += labels.numel()
        total_sum += torch.sum(f_data).item()
        total_sum_sq += torch.sum(f_data ** 2).item()

        positive_movement_directions += (f_data >= 1).sum().item()
        positive_movement_directions_target += (labels >= 0).sum().item()

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - (mean ** 2)
    std = math.sqrt(max(0.0, variance))

    return mean, std, (positive_movement_directions / total_count), (positive_movement_directions_target/total_count_labels)


def print_embeddings(model):
    final_embedding_weights = model.mask_embedding.weight.detach().cpu().numpy() # shape: [categories=5, dim=3]
    print("--------------------------[EMBEDDINGS]---------------------")
    for i in range(final_embedding_weights.shape[0]):
        print(f"Timestep {i}: {np.round(final_embedding_weights[i, :], 3)}")


def print_feature_attentions(model):
    attentions = model.MI_LSTM_layer.global_total_attentions / model.MI_LSTM_layer.total_steps
    features = ["log_return", "rsi", "macd", "vwap", "volume_change"]
    print("--------------------------[ATTENTION]---------------------")
    for i, feature in enumerate(features):
        print(f"{feature}: {np.round(attentions[i], 3)}")


def print_time_and_days_distributions(dataloader):
    fixed_tz = datetime.timezone(datetime.timedelta(hours=1)) # be aware that this is wrong for german timezone -> winter/summer change +01/+02
    # extract all the datetime values of samples in the dataloader
    all_day_names = []
    all_times = []
    for features, _, times in dataloader:
        all_times.extend(features["time_mask"].tolist())
        times_parsed = pd.to_datetime(times, utc=True)
        times_parsed = times_parsed.tz_convert(fixed_tz)
        all_day_names.extend(times_parsed.day_name().tolist())

    unique_days, day_counts = np.unique(all_day_names, return_counts=True)
    unique_times, time_counts = np.unique(all_times, return_counts=True)
    day_counts_relative = day_counts / day_counts.sum()
    time_counts_relative = time_counts / time_counts.sum()

    # plot the distributions
    plt_console.bar(unique_days.tolist(), day_counts_relative.tolist())
    plt_console.title("Days distribution")
    plt_console.xlabel("days")
    plt_console.ylabel("relative frequency")
    plt_console.theme("dark")
    plt_console.show()
    plt_console.clf()
    print("\n")
    plt_console.bar(unique_times.tolist(), time_counts_relative.tolist())
    plt_console.title("Times distribution")
    plt_console.xlabel("timesteps")
    plt_console.ylabel("relative frequency")
    plt_console.theme("dark")
    plt_console.show()
    plt_console.clf()
    print("\n")


def plot_train_test_loss_values(train_loss_vals: list,  test_loss_vals: list, epochs: int):
    print(f"FIRST TRAIN LOSS: {train_loss_vals[0]}")
    print(f"FINAL TRAIN LOSS: {train_loss_vals[-1]}\n")

    print(f"FIRST TEST LOSS: {test_loss_vals[0]}")
    print(f"FINAL TEST LOSS: {test_loss_vals[-1]}")

    plt.plot(range(epochs), train_loss_vals, color='blue', label='Training Loss')
    plt.plot(range(epochs), test_loss_vals, color='orange', label='Test Loss')
    final_train_loss = f"Final Train Loss: {train_loss_vals[-1]:.4f}"
    final_test_loss = f"Final Test Loss: {test_loss_vals[-1]:.4f}"

    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label="Training Loss"),
        Line2D([0], [0], color='orange', lw=2, label="Test Loss"),
        Line2D([0], [0], color='none', label=final_train_loss),
        Line2D([0], [0], color='none', label=final_test_loss)
    ]

    plt.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.savefig(os.path.join("outputs", "training_loss_plot.png"), format='png', dpi=300)
    #plt.show()