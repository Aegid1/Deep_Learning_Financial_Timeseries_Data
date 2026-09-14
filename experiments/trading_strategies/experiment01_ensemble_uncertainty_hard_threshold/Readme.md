### Experiment 1 — can the existing models be used for a simple trading strategy?

The first end-to-end backtest in this repository, deliberately kept as simple as possible: not an
attempt at a competitive strategy, but a sanity check for whether the price-prediction models
trained in [model_architectures](../../model_architectures) carry enough signal to be used for
trading at all, and how they compare to standard baseline architectures on the same task.

- **Models compared:** the repository's own MI-LSTM
  ([Experiment 30](../../model_architectures/experiment30_smooth_l1_loss)) against three common
  SOTA baselines re-implemented in `benchmark_models/` — ALSTM, GAT and GRU. For each architecture, the top-5 checkpoints by test loss are loaded as an ensemble
  (`EnsembleWrapper.py`); at inference time every ensemble member predicts independently and the
  **mean** and **std** across members are used as the point forecast and an uncertainty estimate,
  respectively.
- **Strategy (`benchmark.py`, `apply_trading_strategy`):** The predicted gross return is centered around 0
  (`expected_return = prediction - 1.0`) and reduced by assumed trading costs
  (`net_return = expected_return - costs`). Position size scales with the ratio of net return to
  total uncertainty (market volatility **+** `alpha=1.5` × ensemble disagreement), clipped to
  `[0, w_max=1.0]`. A position is fully closed (`sell_signal`) whenever the expected return drops
  to `<= -0.3%`. This intentionally simple design was the point of this experiment: if it can't
  already beat a passive benchmark with the current models, more elaborate strategy logic is not
  worth building yet.
- **Data:** 11 DAX stocks, test period only (`benchmark_models(stocks_to_benchmark=[...])`), see
  `outputs/<MODEL>/<MODEL>_<Ticker>_preds.csv` for the per-timestep ensemble mean/std, and
  `outputs/analysis/actual_returns.png` for the true return distribution over the same period.

#### Result: the strategy never actually traded on a signal

The reason becomes clear when looking at the raw predictions
(`outputs/analysis/MILSTM_ensemble_preds_mean.png` vs. `outputs/analysis/actual_returns.png`)

Actual test-set returns, spread over roughly **±5-8%** per 2h window across all 11
stocks (see `outputs/analysis/actual_returns.png`), with real heavy-tail outliers. 

MILSTM's predictions is far away from this: across 5016 predictions on the test set, the largest 
prediction is **1.0053%** — roughly 60× smaller than the strategy's own `-0.3%` sell
threshold. In practice, the model always predicts "almost exactly no movement," so `net_return` never clears the cost/uncertainty bar needed to
produce a real buy signal, and the sell threshold is also never reached.

**Conclusion:** none of the tested models predict anywhere near the heavy
tails that actually occur in the data. They collapse toward predicting the mean most
severely so for MILSTM. This isn't a flaw in the backtest or the strategy logic but a flaw in the prediction behavior. 
A follow-up experiment in `model_architectures` should explore ways to push the
model toward predicting the heavy tails specifically.
