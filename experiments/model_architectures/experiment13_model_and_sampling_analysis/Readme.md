### Experiment 13 — does the training-sample distribution bias the model?

Has the same 4-stream MI-LSTM architecture as the [Experiment 10](../experiment10_minimized_features) lineage (stock price, RSI,
MACD, VWAP via LSTM; EMA via DNN). It is trained on two differently-sampled datasets to test the effect of the sampling-bias.
fact:

- `sample_bias/` — trained on `sample_size_25_sample_bias`
- `no_sample_bias/` — trained on `sample_size_23_no_sample_bias`

| seq_size | input_features (LSTM)                    | input_features (DNN) | input normalization                                | loss | optimizer                 | predicted quantity     | evaluation criteria                                |
|:---------|:----------------------------------------------|:--------------------------|:----------------------------------------------------------|:-----|:-----------------------------|:---------------------------|:---------------------------------------------------------|
| 25 / 23  | stock price, RSI, MACD, VWAP (4 streams)       | EMA (1 stream)             | min-max scaling for features; log-transform for the target (`Dataset.py`) | MSE  | AdamW + CosineAnnealingLR   | log(stock price in 2 hrs) | final_test_loss (MSE) + percentage deviation (`utils.train_model`) |

- **Architecture:** same MI-LSTM fusion pattern as Experiment 10, with a proper forget gate
  (`c_t = f_t * c_t + L_t`, `f_t = sigmoid(...)`), sigmoid input gates, tanh candidates, a
  sigmoid-squashed output gate, and `h_t = o_t * tanh(c_t)`. `Model.py` now also takes `seq_size` as an explicit
  parameter (used to size the EMA DNN branch), so it works for both the 25-step and 23-step sample
  sets without a code change.

- **Method:** `analysis.py`/`test_model.py` plot (1) the distribution of weekdays and intraday
  timestamps present in a training sample, and (2) prediction error and directional accuracy,
  grouped by weekday and by time-of-day, on the test set — generated per-variant into
  `sample_bias/outputs/` and `no_sample_bias/outputs/`.

- **Finding:** prediction error and direction accuracy correlate strongly with the time-of-day and weekday distribution 
of the training samples. The model performs measurably worse at times that were underrepresented during training. The 
performance increased for the model which was trained on a more uniform distribution of the times.

- **Reason for sample bias**: The sample bias occurred due to the used sequence_size of the sampled timeseries. The raw data is sampled based on a
2-hour frequency and thus there exist 5 samples for each day and 25 samples for each week. Initially these raw samples 
were processed further to training samples, where a training sample consists of a timeseries of 25 records. These training samples
where created in a tumbling-window manner to mitigate data leakage.
So when the first window started for example at Monday 09:00 (=first timestep) this resulted to creating training samples that always had the target
to predict the stock price for the next Monday 09:00. This kept going because of using always a whole week of records.
Thus, it was later used a sequence size of 23 such that there is more variation in the time of the target.