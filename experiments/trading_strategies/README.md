### Trading strategies

Experiments that turn model predictions into trading signals and backtest them (via `vectorbt`).

- `benchmark_models/` — code only: SOTA baseline architectures (ALSTM, GAT, GRU, Localformer, TCN,
  Transformer) used as benchmarks, plus their shared `Dataset.py`/`train.py`.
- `trained_benchmark_models/` — artifacts only: trained weights produced by `benchmark_models/train.py`,
  namespaced per feature set.
- `simulation.py` — shared backtest/visualization logic reused across experiments.
- `papers/` — relevant or implemented papers.
- `experimentXX_*/` — one trading strategy per experiment, each defining its own signal logic
  (e.g. `apply_trading_strategy`) and pointing at a specific `trained_benchmark_models/<feature_set_name>/`.
