### Experiment 10 — does a smaller feature set hurt?

Reduces the 11-feature MI-LSTM down to a minimal 5-feature version (stock price, RSI, MACD, VWAP via LSTM; EMA via DNN),
dropping MACD-signal, OBV, SMA and the Bollinger Bands as these are the most correlated features based on the results of 
`experiment6_feature_analysis` in `feature_engineering`.

Testing if the model performs the same with less features available.

| seq_size | input_features (LSTM)              | input_features (DNN) | input normalization                        | loss | optimizer            | predicted quantity   | evaluation criteria                               |
|:---------|:----------------------------------------|:--------------------------|:------------------------------------------------|:-----|:-----------------------|:-------------------------|:-------------------------------------------------------|
| 25       | stock price, RSI, MACD, VWAP (4 streams) | EMA (1 stream)             | min-max scaling, fit per training batch (`train.py`) | MSE  | SGD (momentum=0.9)     | stock price in 2 hrs    | final_test_loss (MSE) + a std/percentage-deviation metric |

- **Architecture (high-level):** Same MI-LSTM fusion pattern as in previous experiments, with the DNN branch
  reduced to just EMA and the LSTM branch reduced to 4 streams — `Model.py` is otherwise unchanged
  architecturally.

- **Hyperparameters:** batch_size=32, hidden_size=50, lr=0.001, epochs=100

**NOTE:**
This experiment was still trained on the training samples that contain the sample bias which was identified in 
`experiment12_model_and_sampling_analysis` and fixed in the later experiments.