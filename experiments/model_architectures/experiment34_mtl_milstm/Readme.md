### Experiment 34 — full MI-LSTM backbone + multi-task heads

The capstone of the multi-task line: combines the Multi-Input-LSTM backbone from Experiments
27-30 (5 parallel per-feature LSTMs fused via `MultiInputLSTMLessLSTMs`, plus a learned
time-of-day embedding and a separate EMA branch) with the per-task-head, uncertainty-weighted
multi-task setup from Experiment 33 — the first MTL experiment in this series to use the full
technical-indicator feature set rather than 1-3 features.

| seq_size | input_features                                                          | input normalization                                                    | loss                                              | optimizer                 | predicted quantities                                     | evaluation criteria |
|:---------|:----------------------------------------------------------------------------|:-----------------------------------------------------------------------------|:------------------------------------------------------|:----------------------------|:---------------------------------------------------------------|:-----------------------|
| 23       | stock-return, log-volume-change, RSI, MACD, VWAP, EMA, time-of-day mask | min-max scaling for RSI/MACD/VWAP/log-volume-change; time-mask via learned embedding | SmoothL1Loss + MSE + MSE + BCE, uncertainty-weighted (4 task heads) | AdamW + CosineAnnealingLR | next-period return, log-volume-change, RSI, return direction | final_test_loss + correct_direction; best-of-N kept |

- **Architecture (high-level):** 5 features each pass through their own LSTM (conditioned on the
  time-of-day embedding, as in Experiments 29/30), fused via the MI-LSTM attention layer; the EMA
  branch and MI-LSTM output are concatenated and passed through a shared DNN before splitting into
  4 independent task heads (return, log-volume-change, RSI, direction), combined via the same
  uncertainty-weighted loss as Experiment 33.

- **Results:** `results.csv` (13 runs). Best `final_test_loss` ≈ **0.483**, `correct_direction`
  values range ~0.47-0.55 across runs — in the same barely-above-chance range as Experiments 32/33,
  i.e. adding the full feature set back did not produce a clear improvement in direction accuracy
  over the simpler 3-feature multi-task models.
