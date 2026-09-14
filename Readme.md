<div align="center">
  
  # 📈 Deep Learning for Stock Price Movement Prediction
  
  **A modular repository containing multiple experiments in feature engineering, specific LSTM-based model architectures and backtesting
  with Vectorbt.**

  [![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](#)
  [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](#)
  [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](#)
  [![VectorBT](https://img.shields.io/badge/vectorbt-Backtesting-success.svg)](#)

   Data Retrieval - PreProcessing - Model Training - Backtesting - Debugging

</div>

---

## 💡 Executive Summary / TL;DR

This repository explores the usefulness of recurrent neural networks for predicting intraday stock price movements. 
In addition to pure price predictions, the focus is on practical applicability through direct backtesting of the model 
signals in real-world trading scenarios.
> **Key Result:**

---

## 🎓 Background

This project started out of a university project and was originally intended to get familiar with
deep learning architectures (particularly LSTM variants). The experiment history reflects this iterative and
exploratory approach. All training has been performed exclusively on the CPU of
a personal laptop and GPU training has not yet been considered. 

In the most recent
experiments, the focus has shifted from pure architecture exploration to actual
model performance and an initial backtesting simulation.

---

## 📊 Further Key Results
In the following the key results are shown for predicting intraday stock price movement on market timing.

### Relevant Features 
<p align="center">
  <img src="https://via.placeholder.com/800x350/1e1e24/00d2ff?text=VectorBT+Equity+Curve+(Portfolio+Value+vs.+Baseline)" alt="placeholder" width="800"/>
</p>

### Strategy-Performances (Backtests with VectorBT)

<p align="center">
  <!-- Platzhalter für dein VectorBT Chart -->
  <img src="https://via.placeholder.com/800x350/1e1e24/00d2ff?text=VectorBT+Equity+Curve+(Portfolio+Value+vs.+Baseline)" alt="placeholder" width="800"/>
</p>

---

## 🧠 Implemented model-architectures (PyTorch)

The repository contains implementations of the following deep learning architectures:

- [x] **SOTA Deep-Learning Models:** baseline models that are commonly used for this scenario (Vanilla-LSTM, GRU, GAT, ALSTM)
- [x] **Attention-based LSTM**
- [x] **Multi-Task Learning LSTM**

---

## 📁 Repository Structure

```text
├── images/                     
├── experiments/
│   ├── feature_engineering/            # see feature_engineering/README.md
│       ├── experimentXX_*
│   ├── model_architectures/            # see model_architectures/README.md
│       ├── experimentXX_*              # containing the same structure as above
│           ├── outputs/                # train test results
│           ├── Dataset.py              # pre processing of train test samples
│           ├── Model.py                # definition of general model architecture
│           ├── custom_layers.py        # definition of more complex architecture layers 
│           ├── train.py                # training initialization
│           ├── analysis.py             # calculation and visualization of training metrics
│   └── explainable_ai/                 # see explainable_ai/README.md
│       ├── experimentXX_*
│   └── trading_strategies/             # see trading_strategies/README.md
│       ├── benchmark_models/           # SOTA baseline model code (no trained weights)
│       ├── trained_benchmark_models/   # trained weights, namespaced per feature set
│       ├── experimentXX_*
├── utils/                        
│   ├── experiment_utils.py             # train and test loop related util functions
│   ├── macro_factor_utils.py           # generation of macro economic features 
│   ├── news_data_utils.py              # generation of news-related features
│   ├── stock_data_utils.py             # retrieval and handling of raw stock data
│   └── technical_indicators_utils.py   # generation of technical indicators