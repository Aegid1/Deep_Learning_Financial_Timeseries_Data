### Model architectures

Research on the repository's own (non-SOTA-baseline) model architectures — LSTM-based and
multi-task variants — and their training procedure.

- `papers/` — relevant or implemented papers.
- `experimentXX_*/` — one architecture/training iteration per experiment, each self-contained:
  - `Dataset.py` — pre-processing of train/test samples
  - `Model.py` — model architecture definition
  - `custom_layers.py` — more complex architecture layers
  - `train.py` — training initialization
  - `analysis.py` — calculation and visualization of training metrics
  - `outputs/` — train/test results, incl. trained weights for that experiment

Each experiment duplicates rather than shares this code on purpose: it keeps every past
experiment independently reproducible, even as later experiments evolve the architecture.

**Numbering:** experiment numbers are not contiguous (10, 13, 14, then 25-34). The missing experiments
were removed from this repository and not lost. Each gap exists for one of the following reasons: the
experiment became outdated, the methodology used wasn't clear and strict enough to draw a meaningful conclusion from 
or the result turned out to be trivial in retrospective and wasn't worth keeping as a folder.

- **Experiments 1-9:** mostly about feature *normalization*, so testing different ways to scale
  inputs, manually experimenting with hyperparameter configurations, and iterating on the general training
  procedure rather than the model architecture itself.
- **Experiments 14-25:** the prediction target changed from the absolute stock price to the
  log-return (the convention Experiments 27+ still use). Different `seq_size` values were tested
  manually here too, but not systematically analyzed. These experiments also tested whether the number of
  news articles behind a sentiment score adds information beyond the sentiment score itself.