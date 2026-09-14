### Experiment 33 — separate output heads per task

Same 4-task setup as Experiment 32 (return, log-volume-change, RSI, direction; same inputs, same
seq_size=23), but the single shared `Linear(hidden_size, 4)` output layer is replaced with one
independent `Linear` head per task — "more task heads" in the folder name refers to this
architectural change, not to adding more tasks.

| variant                            | task setup                                                                                   | loss                                              | best final_test_loss (typical range) |
|:-------------------------------------|:--------------------------------------------------------------------------------------------------|:-----------------------------------------------------|:------------------------------------------|
| `MultiTask`                          | 4 independent heads, unweighted average loss                                                  | SmoothL1Loss + MSE + MSE + BCE, averaged            | ~0.55-0.65 (18 runs)                       |
| `MultiTaskWithLossWeights`           | 4 independent heads, uncertainty-weighted loss (same scheme as Experiments 31/32)             | as above, uncertainty-weighted                       | ~0.51-0.63 (34 runs)                       |
| `MultiTaskWithLossWeightsAdjusted`   | uncertainty weighting **plus** each task's loss normalized by its own initial magnitude, with the return task additionally upweighted 10x | as above, normalized + uncertainty-weighted          | ~0.50-0.73 (16 new runs, see note)         |

- **Architecture (high-level):** Single-layer LSTM over the 3 input features, feeding four
  separate linear heads (`stock_return_head`, `rsi_head`, `log_volume_change_head`,
  `return_direction_head`). `MultiTaskWithLossWeightsAdjusted` (`Model.py`) additionally records
  each task's mean loss on the very first batch (`self.init_losses`) and divides all subsequent
  losses by it before applying the uncertainty weights — intended to stop whichever task starts
  with the largest raw loss from dominating the combined objective.
- One can further see that during the training of `MultiTaskWithLossWeights` the gradient norms 
start to increase at a given point rather than decreasing until the end. This makes the whole training unstable.
- This is mainly because of the formulation of the paper where the total loss is defined as:
$$\mathcal{L}_{orig} = \frac{1}{2} e^{-s} \mathcal{L}(W) + \frac{1}{2} s$$
for which the gradient looks like this
$$\frac{\partial \mathcal{L}_{orig}}{\partial s} = -0.5 e^{-s} \mathcal{L}(W) + 0.5$$
So when the loss is very small, like 10^-3, then the gradient remains almost constant around 0.5, which results to large 
updates of s. This leads to destabilizing the optimization which led to the phenomenon of increasing gradient norm mentioned before.
- Using the proposed normalization of `MultiTaskWithLossWeightsAdjusted` stabilized the gradient norms and the training
atleast in some cases. The rough idea behind this is to make the gradient to s dependent on some quantity that decreases
during training, for which in this case the loss itself was chosen:
$$\frac{\partial \mathcal{L}_{mod}}{\partial s} = \mathcal{L}(W) \cdot \left( -0.5 e^{-s} + 1 \right)$$
Which leads to the overall loss formulation of:
$$\mathcal{L}_{mod} = \mathcal{L}(W) \cdot \left( \frac{1}{2} e^{-s} + s \right)$$

- This leads to smoother optimization dynamics, however also destroys the core idea of the paper when setting
the updated gradient to zero:
$$\mathcal{L}(W) \cdot \left( -0.5 e^{-s} + 1 \right) = 0$$
Because the loss cant be zero in practice the optimum for the task weights can only be achieved when the term in the brackets
becomes zero
$$-0.5 e^{-s} + 1 = 0 \implies 0.5 e^{-s} = 1 \implies e^{-s} = 2 \implies s = -\ln(2) \approx -0.693$$
This results to a constant task weight which is not dependend on the loss of the task as for the original formulation:
$$\frac{\partial \mathcal{L}_{orig}}{\partial s} = -0.5 e^{-s} \mathcal{L}(W) + 0.5$$
$$-0.5 e^{-s} \mathcal{L}(W) + 0.5 = 0 \implies e^{-s} = \frac{1}{\mathcal{L}(W)}$$

- In order to achieve numeric stability and to keep the idea of the original paper this has to be further analyzed and enhanced later!
