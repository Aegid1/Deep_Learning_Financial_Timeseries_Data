### Experiment 7 — does global standardization beat per-window min-max scaling?

Motivation: the feature-distribution analysis in
[Experiment 2](../experiment2_feature_distributions), which proposes a global, std-based scaling
(with outlier clipping) per feature, and the **underfitting problem** observed in the model
experiments.
Models end up predicting very conservative and their predictions have very low variance and collapse toward a near-constant
value. Per-window min-max scaling removes the absolute level of each feature in every sample, so the
hypothesis was that a global scaling that preserves magnitudes lets the model have more information and therefore might 
make more confident predictions.

Both variants train the same plain LSTM (`Model.py`, `SimpleLSTM`, all features concatenated into
one input of size 7) on the same sample set (`sample_size_23_stationary_volume_new`, 23-step
windows, target = log-return) with MSE loss, AdamW and cosine annealing, and use **the same feature
set** (log_return, time_mask, rsi, macdhist, vwap_p, log_volume_change, ema_p). Only the input
scaling differs:

| variant             | scaling (`Dataset.py`)                                                                                                         |
|:--------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| `min_max_scaling/`  | per-window min-max for `rsi`, `macdhist`, `vwap_p`, `log_volume_change`; `log_return` and `ema_p` passed in unscaled           |
| `standardization/`  | global, from the training set: transform (`rsi`: `(x-50)/50`; `vwap_p`, `ema_p`: `log`) → clip outermost 0.5% → divide by std |

- **Metrics:**
  - *percentage deviation* — mean absolute difference between predicted and actual gross return
    mean(`exp(prediction) - exp(target)`) × 100.
  - *std coefficient* (`test_model_predictions_std`) — std of the predictions divided by std of the
    true targets on the benchmark test data. 
  
    ≈ 1: the model's predictions are as spread out as
    reality;
    
    ≈ 0: very conservative; 

    ≫ 1: overconfident.


#### Results (10 independent runs  mean ± std over runs)

| variant                     | percentage deviation | std coefficient (min – max)  | duration per run |
|:----------------------------|:----------------------|:------------------------------|:------------------|
| `min_max_scaling/`          | 0.543 ± 0.028         | 0.083 ± 0.011 (0.07 – 0.10)   | ~82 s             |
| `standardization/`          | 0.544 ± 0.031         | 0.117 ± 0.031 (0.04 – 0.15)   | ~28 s             |
| baseline: always predict 0  | not recorded yet (`baseline.py`) | 0 by definition     | –                 |

- **Accuracy: identical.** 0.543 vs. 0.544 (Welch t ≈ 0.05); the run-to-run std (~0.03)
- **Prediction spread: standardization is modestly higher, but both are very conservative.** 0.117
  vs. 0.083 (+40%, Welch t ≈ 3.3) — a real but small difference. Both variants predict only about a
  tenth of the spread of the real targets
- **Training time reflects the data pipeline, not the model.** The min-max runs take ~82 s with
  52k parameters (and took ~70 s with only 5k in an earlier configuration) while the
  standardization runs take ~28 s with 22k — the run time hardly depends on model size, which points
  to the per-sample `MinMaxScaler` fitting in the min-max `Dataset` as the dominant cost.

#### What the results say

Global standardization is the better input pipeline with being 3× faster and achieving bit more spread. It does not
make the models more accurate, and it does not solve the underfitting problem.