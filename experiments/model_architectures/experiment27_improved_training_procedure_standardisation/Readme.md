### Experiment 27

| seq_size | input_features                                                                  | input normalization                                                             | loss | optimizer         | predicted quantity           | evaluation criteria                          |
|:---------|:---------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:-----|:-------------------|:-------------------------------|:------------------------------------------------|
| 23       | log-return, log-volume-change, RSI, MACD, VWAP, oil_price, eur_usd, EMA | z-score standardization (train-set mean/std, with 1.5σ outlier clipping on MACD and log-return) | MSE  | AdamW + CosineAnnealingLR | next-period stock log-return | final_test_loss (MSE) + percentage deviation; best-of-10 fresh initializations kept |

- **Architecture (high-level):**
Multi-Input LSTM (MI-LSTM): 7 of the features each pass through their own single-layer LSTM + LayerNorm, then get fused via an attention-gated `MultiInputLSTMLessLSTMs` layer (`custom_layers.py`). EMA is processed separately through a small DNN branch. Both outputs are concatenated and passed through two final DNN layers to produce the return prediction.

- **Results:** Two checkpoints saved (`0.5248`, `0.5482` percentage deviation) — this run predates the `results.csv` logging used in later experiments, so per-run metrics aren't tabulated here.

- **Notes:**
  - `train.py` has the feature-standardization call (`dataset.calculate_feature_mean_std(train_loader)`) commented out. Re-running the script in this state would not reproduce the two saved checkpoints, which were produced by an earlier, since-edited version of this file.
