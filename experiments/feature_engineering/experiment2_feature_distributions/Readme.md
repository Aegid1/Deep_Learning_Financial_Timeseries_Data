### Experiment 2 — what do the input features look like, and how should they be scaled?

This experiment looks at the empirical distribution of every input feature in the training samples and derives a scaling 
per feature. The motivation is optimization as the inputs have different scales that make the loss surface badly conditioned 
as gradients for the large-scale features dominate the optimization and the training becomes slow and unstable. 
Bringing every feature to a comparable scale with roughly a zero mean and a unit variance reduces the condition number 
of the problem and thus simplifies the optimization procedure. This experiment only defines and visualizes the scaling but 
whether it actually helps a model is tested in [Experiment 7](../experiment7_new_scaling).

- **Method (`analysis.py`):** all training samples of one sample set (e.g.
  `sample_size_23_stationary_volume_new`) are loaded and concatenated per feature.
  The function `plot_distributions_of_features` clips the outermost 0.5% of every feature and
  applies the candidate scaling and plots a histogram + a KDE with the resulting mean and std.
  The functions `compute_oil_change` / `compute_currency_change` do the same for the macro features (5-step
  log-change of oil price and EUR/USD, divided by its maximum absolute value).
- **Scaling rules:**

| feature                                  | scaling                                  | assumption behind it                                                |
|:-----------------------------------------|:-----------------------------------------|:--------------------------------------------------------------------|
| `log_return`, `log_volume_change`, `macdhist` | divide by the feature's std         | already centered around 0 (mean ≈ 0), so no mean subtraction needed |
| `rsi`                                    | `(rsi - 50) / 50`, then divide by std    | bounded in [0, 100] with a natural center at 50                     |
| `vwap_p`, `ema_p`                        | `log(x)`, then divide by std             | price-relative ratios centered at 1.0 → the log centers them at 0   |

  The std is a single global value per feature which is computed over the whole training set and not per
  sample/window.

#### What the distributions show

- Plots in this folder: `feature_distributions.png` (raw, after clipping) vs.
  `feature_distributions_post_transformations.png` (after the scaling above).
- **Heavy tails:** `log_return`, `macdhist`, `vwap_p` and `ema_p` are sharply peaked with long
  outliers (raw `vwap_p` reaches ~2.0 while its std is only ~0.012). Clipping the outer 0.5% plus the
  std scaling brings them into roughly [-5, +5].
- **`rsi`** is already well-behaved (bell-shaped, std ≈ 12.5) and only needs centering/scaling.
- **`log_volume_change` is trimodal:** besides the main peak around 0 there are two separate modes at
  about ±11 (raw), which remain at about ±1.6 after scaling. Std scaling normalizes the spread but
  does not remove this structure; the cause of the side modes was not investigated in this experiment but is probably
  because of the daily cycle.
- **Macro features** (`oil_change_distribution.png`, `currency_change_distribution.png`): almost all
  mass sits in a very narrow band around 0 with fat, bumpy tails. Dividing by the maximum absolute
  value, as done here, lets those outliers set the scale and squeezes the bulk of the data close to
  0 — the 0.25% / 99.75% percentile lines (red) lie far outside the region where nearly all
  values are. This is the main argument for std scaling + clipping over max-abs scaling.
