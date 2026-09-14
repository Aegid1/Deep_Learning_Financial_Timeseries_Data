### Experiment 26

| seq_size | input_features                                                                                                     | input normalization           | loss                   | optimizer | predicted quantity      | evaluation criteria               |
|:---------|:-------------------------------------------------------------------------------------------------------------------|:------------------------------|:-----------------------|:----------|:------------------------|:----------------------------------|
| 23       | log-return, EMA, RSI, MACD, VWAP, log-volume-change, eur_usd, oil_price, stoch_k, stoch_d, will_r, adosc, cci, mom | min-max scaling (batch-based) | BinaryCrossEntropyLoss | SGD       | stock movement in 2 hrs | true stock movement vs. predicted |

- **Architecture (high-level):**
Oil price, currency and volume technical indicators are processed in single LSTMs and single DNNs. 
Other features are processed by single LSTMs and processing these results in a single Multi-Input-Attention-LSTM.
The EMA is processed through a single DNN. Processing these outputs together with two further DNNs.