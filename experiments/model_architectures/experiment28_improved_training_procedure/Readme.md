### Experiment 28

| seq_size | input_features                                                | input normalization                                                                  | loss | optimizer | predicted quantity           | evaluation criteria                          |
|:---------|:-----------------------------------------------------------------|:-----------------------------------------------------------------------------------------|:-----|:-----------|:-------------------------------|:------------------------------------------------|
| 23       | log-return, RSI, MACD, EMA, VWAP, log-volume-change | min-max scaling (batch-based) for RSI/MACD/EMA/VWAP/log-volume-change; log-return left unscaled | MSE  | AdamW      | next-period stock log-return | final_test_loss (MSE) + percentage deviation; best-of-10 fresh initializations kept |

- **Architecture (high-level):**
Same Multi-Input LSTM backbone as Experiment 27 (per-feature LSTMs fused via `MultiInputLSTMLessLSTMs`, EMA processed via a separate DNN branch), but the shared `utils.experiment_utils.train_model` call was refactored to take the raw `dataset` + batch size directly (instead of pre-built train/test `DataLoader`s) and to no longer standardize features by mean/std — this experiment is about the *training procedure* itself, not a new architecture.

- **Results:** `results.csv` (10 fresh-initialization runs). Best `percentage deviation` ≈ **0.485** (hidden_size=50, lr=2e-4).
