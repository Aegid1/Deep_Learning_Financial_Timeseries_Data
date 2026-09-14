### Experiment 25

Based on previous experiments the proposed features performed the best based on the chosen evaluation criteria. The purpose 
of this experiment is to apply multi-task-learning with these features.

| seq_size | input_features                                                          | input normalization           | loss                   | optimizer | predicted quantity   | evaluation criteria                                                              |
|:---------|:------------------------------------------------------------------------|:------------------------------|:-----------------------|:----------|:---------------------|:---------------------------------------------------------------------------------|
| 23       | log-return, EMA, RSI, MACD, VWAP, log-volume-change, eur_usd, oil_price | min-max scaling (batch-based) | MSE / CrossEntropyLoss | SGD       | stock price in 2 hrs | mse of true log stock return vs. predicted and true stock movement vs. predicted |

- **Architecture (high-level):**
Oil price and currency are processed in single LSTMs with seq_size = 4 and single DNNs. 
Other features are processed by single LSTMs and processing these results in a single Multi-Input-Attention-LSTM.
The EMA is processed through a single DNN. Processing these outputs together with two further DNNs.

3 more DNNs are used for processing these outputs for the stock return prediction and 3 more DNNs for predicting the log return.