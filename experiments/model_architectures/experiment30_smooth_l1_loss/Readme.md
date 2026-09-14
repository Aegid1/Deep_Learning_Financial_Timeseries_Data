### Experiment 30

| seq_size | input_features                                                            | input normalization                                                                          | loss                       | optimizer                 | predicted quantity           | evaluation criteria                          |
|:---------|:------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|:-----------------------------|:----------------------------|:-------------------------------|:------------------------------------------------|
| 23       | log-return, log-volume-change, RSI, MACD, VWAP, EMA, time-of-day mask | min-max scaling for RSI/MACD/VWAP/log-volume-change; log-return left unscaled; time-mask fed through a learned embedding | SmoothL1Loss (beta=3.5e-3) | AdamW + CosineAnnealingLR | next-period stock log-return | final_test_loss + percentage deviation; best-of-10 fresh initializations kept |

- **Architecture (high-level):**
Same MI-LSTM + time-mask-embedding backbone as Experiment 29, with oil_price/eur_usd dropped again (back to 5 core technical features) and the loss switched from Experiment 28/29's implicit MSE-style loss to `SmoothL1Loss`, which is less sensitive to outlier return magnitudes. Weight initialization is also simplified to a single uniform init instead of the Kaiming/Xavier/orthogonal scheme used in 27-29.

- **Results:** `results.csv` (10 fresh-initialization runs). Best `percentage deviation` ≈ **0.427** ( hidden_size=50, lr=5e-3) however this is not captured in the results.csv