### Experiment 14 — feeding news-sentiment features into the MI-LSTM

Extends the [Experiment 13](../experiment11_minimized_features)
MI-LSTM lineage with two new sequential features derived from news sentiment —
`short_term_impact` and `long_term_impact` — the same features whose correlation with future stock
returns was measured in [`feature_engineering/experiment6_feature_analysis`](../../feature_engineering/experiment6_feature_analysis).
This is the first architecture experiment to actually consume that feature-engineering result as a
model input, rather than just analyzing it.

| seq_size | input_features (LSTM)                                                          | input_features (DNN) | input normalization              | loss | optimizer                 | predicted quantity   | evaluation criteria                                |
|:---------|:-------------------------------------------------------------------------------------|:--------------------------|:---------------------------------------|:-----|:-----------------------------|:-------------------------|:---------------------------------------------------------|
| 25       | stock price, RSI, MACD, VWAP, short_term_impact, long_term_impact (6 streams)         | EMA (1 stream)             | min-max scaling (`Dataset.py`)          | MSE  | AdamW + CosineAnnealingLR   | stock price in 2 hrs    | final_test_loss (MSE) + percentage deviation (`utils.train_model`) |

- **Architecture (high-level):** Same MI-LSTM fusion pattern as Experiments 11/12
  (`ModelLessLSTMs` + `MultiInputLSTMLessLSTMs`), extended from 4 to 6 LSTM-routed streams to make
  room for the two sentiment-impact features. `custom_layers.py`'s `MultiInputLSTMLessLSTMs` layer
  has been updated accordingly: two new input-gate parameter sets (`W_i_sti`/`W_i_lti`) were added
  for the new streams, they share the existing candidate-weight (`W_c_n`), output-weight (`W_o`)
  and attention-weight (`W_a`) parameters with RSI/MACD/VWAP (matching how those three already
  shared them), and the attention softmax now runs over 6 streams instead of 4. No other part of
  the architecture changed.

- **Hyperparameters (intended, see below):** sample_size_stocks=23, batch_size=32, hidden_size=50,
  lr=0.005, epochs=100, trained on the `sample_size_23_news_combined` sample set.
