# PGA-Trans-HAR: Prior-Guided Attention Spatio-Temporal Transformer for Global Stock Market Volatility Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of the paper: **"Prior-Guided Attention Spatio-Temporal Transformer for Global Stock Market Volatility Forecasting"**.

---

## 📖 Overview
**PGA-Trans-HAR** is a hybrid deep learning and econometric framework designed for global multi-market realized volatility (RV) forecasting. It bridges modern Spatio-Temporal Transformers with classical Heterogeneous Autoregressive (HAR) models, introducing **Prior-Guided Attention (PGA)** and **Attention Padding Masks** to address key empirical challenges in financial time-series forecasting:
1. **Non-Common Trading Days (Holiday Mismatches)**: Elegantly aligns asynchronous international trading calendars using softmax-based attention masking without recurrent zero-padding noise.
2. **Spatial-Temporal Non-Linearity**: Replaces sequential recurrent bottlenecks (e.g., DCGRU) with parallelizable spatio-temporal self-attention.
3. **Econometric Prior Regularization**: Integrates Diebold-Yilmaz (2012) Vector Autoregression (VAR) GFEVD prior graphs into spatial attention via adaptive node-specific gating to prevent overfitting against high-frequency market noise.

---

## 📂 Project Structure
```text
PGA-Trans-HAR/
│
├── data/
│   └── df_union_sqrt.csv          # Local high-frequency realized volatility dataset
│
├── models/                        # Core model architectures
│   ├── PGA-TRANS-HAR (including experimental version of module ablation).ipynb              # PGA-TRANS-HAR Structure with Outcome
│   ├── PGA-TRANS-HAR (including experimental version of module ablation).py                     # PGA-TRANS-HAR Structure
│   
│
│
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation