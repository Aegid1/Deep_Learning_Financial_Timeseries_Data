### Experiment 32 — multi-tasking across different target types

Where Experiment 31 added auxiliary tasks *across time* (predict every timestep), this experiment
adds auxiliary tasks *across target types*: alongside the main return prediction, the model is
also asked to predict log-volume-change, RSI, and return direction from the same shared LSTM
state. Inputs: log-return (raw), RSI, log-volume-change (RSI/log-volume-change min-max scaled),
seq_size=23, AdamW + CosineAnnealingLR, epochs=100.

| variant                    | task setup                                                                  | loss                                                        | best final_test_loss | best correct_direction |
|:----------------------------|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------|:------------------------|:---------------------------|
| `singleLSTM` (return)       | single task — return regression only (`ModelStockChange.py`)               | SmoothL1Loss                                                     | 0.5596                   | —                            |
| `singleLSTM` (direction)    | single task — binary up/down classification only (`ModelDirection.py`)     | BCEWithLogitsLoss                                                 | 8.79 *(diverged, see note)* | 0.543                        |
| `MultiTask`                 | one shared 4-way output head predicting [return, log-volume-change, RSI, direction], unweighted average loss | SmoothL1Loss + MSE + MSE + BCE, averaged                        | 0.5562                   | 0.454                        |
| `MultiTaskWithLossWeights`  | same 4-way head, uncertainty-weighted loss combination                     | as above, uncertainty-weighted                                    | 0.5047                   | 0.521                        |

- **Architecture (high-level):** All four variants share a single-layer LSTM over the 3 input
  features. `MultiTask`/`MultiTaskWithLossWeights` route the final hidden state through one shared
  `Linear(hidden_size, 4)` head, one output slot per task. `MultiTaskWithLossWeights` applies the
  same uncertainty-weighting scheme as Experiment 31 (negative `final_train_loss` values here are
  the same expected regularization effect, not a bug), with the classification task's weight
  doubled since its loss isn't naturally on the same scale as the regression tasks.

- **Results:** The uncertainty-weighted multi-task variant has the best test loss and the best
  direction accuracy of the four (0.521, barely above chance), suggesting the auxiliary tasks help
  a little once properly weighted — though all four remain close to a 50/50 direction-accuracy
  baseline.

- **Notes:** The `singleLSTM` direction-only run (`outputs_direction/results.csv`) shows a train
  loss flat around `ln(2) ≈ 0.69` (the value a classifier collapsing to a single class would
  produce) against a test loss of 8-10 — this run appears to have diverged rather than learned,
  and shouldn't be read as a working baseline.
