# PGA-Trans-HAR

Official implementation for the preprint paper:  
**Forecasting Global Volatility Across Asynchronous Markets: Incremental Accuracy from Constrained Cross-Market Attention**

**Authors:** Xinlin Zhao, Haotian Qiao
**Repository:** [https://github.com/cillinzhao/PGA-Trans-HAR](https://github.com/cillinzhao/PGA-Trans-HAR)

---

Official PyTorch implementation of the preprint:  
**"Decoding Global Volatility Spillovers: A Neuro-Econometric Spatio-Temporal Transformer for Asynchronous Financial Markets"**  
*Xinlin Zhao (Independent Researcher) and Haotian Qiao (University of Michigan, Ann Arbor)*  
Contact: `cillinzhao@gmail.com` | `qhaotian@umich.edu`

---

## 📌 Abstract & Overview

Forecasting multi-market realized volatility across international equity exchanges faces an inherent **asynchronous measurement challenge**: heterogeneous national holiday calendars create missing observations that disrupt temporal continuity. Naively truncating data to common trading days discards valid economic observations, whereas zero-filling closures creates spurious near-zero volatility signals and distorts cross-market spillover topologies.

**PGA-Trans-HAR** resolves this bottleneck via a novel **neuro-econometric fusion** paradigm that strictly respects the forecast-origin information filtration ($o = t - h$):
1. **Dynamic Econometric Prior:** A rolling Ridge-regularized VAR($p=2$) and Generalized Forecast Error Variance Decomposition (GFEVD, $H_g=22$) network is refreshed every $K_g=20$ origins over trailing window $W_g \le 252$ without look-ahead bias.
2. **Asymmetric Source Masking:** Inactive markets are masked out as temporal/spatial key-value sources while retaining their query states, allowing closed markets to receive contemporaneous shocks without transmitting artificial closure signals.
3. **Adaptive Convex Attention-Prior Gate:** A learned, market-specific gate $g_n = \sigma(\gamma_n)$ dynamically weights data-driven spatial self-attention and the rolling econometric prior.
4. **Frozen HAR Knowledge Anchor:** Neural corrections are bounded via an inverse-softplus formulation:
   $$\hat{x}_{t,n}^{(h)} = \text{softplus}\left( \text{softplus}^{-1}(\hat{x}_{t,n}^{\text{HAR},(h)}) + \alpha_n r_{b,n} \delta_{b,n} \right)$$
   guaranteeing mathematical positivity and anchoring training stability.
5. **Node-Balanced Regularization:** Minimizes node-standardized MSE combined with a smooth-worst HAR-relative regret penalty ($\lambda=0.10, \tau=0.20$) to guard against isolated market deterioration.


---

## 📂 Repository Structure

```text
PGA-Trans-HAR/
├── ablation/                 # Structural ablation implementations
│   ├── pga_ablation.py           
├── baselines/                # External econometric & deep learning baselines
│   ├── baseline_v5_hn.py              
├── data/                     # Dataset and union calendar processing
│   ├── df_union_sqrt.csv     # Pre-transformed percentage square-root RV panel
├── models/                   # Core proposed architecture
│   ├── pga_trans_har.py     
├── .gitignore
├── README.md
└── requirements.txt
