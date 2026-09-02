# PGA-Trans-HAR

Official implementation for the preprint paper:  
**Forecasting Global Volatility Across Asynchronous Markets: Incremental Accuracy from Constrained Cross-Market Attention**

**Authors:** Xinlin Zhao, Haotian Qiao, Ziyao Lin  
**Repository:** [https://github.com/cillinzhao/PGA-Trans-HAR](https://github.com/cillinzhao/PGA-Trans-HAR)

---

## 📌 Overview

Multivariate volatility forecasting across international equity markets presents a fundamental **information-set problem**: asynchronous exchange closures dictate which market observations strictly belong to the information filtration at any given forecast origin $o = t - h$.

**PGA-Trans-HAR** is a parsimonious, econometrically regularized forecasting system designed to extract genuine incremental predictive accuracy over the univariate HAR benchmark without unconstrained overfitting.

### Key Methodological Components:
1. **Forecast-Origin-Admissible Information Alignment:** Formulated on a union calendar ($T = 4,079$ days) distinguishing observed trading days from exchange closures ($m_{t,n} \in \{0, 1\}$). All rolling inputs and graph priors are strictly indexed by the forecast origin $o = t - h$.
2. **Origin-Admissible Predictive-Connectedness Prior:** A rolling ridge-VAR($p=2$) / GFEVD($H_g=22$) directional connectedness prior, dynamically refreshed every $K_g = 20$ forecast origins over a trailing window $W_g \le 252$.
3. **Constrained Cross-Market Attention:** Combines data-driven spatial self-attention $A^D$ and the rolling GFEVD prior $\tilde{P}$ via a learned, time-invariant market-specific gating vector $g_n = \sigma(\gamma_n)$.
4. **Asymmetric Masking Mechanism:** Prevents closed exchanges from acting as temporal/spatial key-value sources (avoiding spurious zero-volatility transmission) while retaining them as query destinations.
5. **Frozen Direct-Horizon HAR Baseline Anchor:** Residual neural learning bounded in the inverse-softplus domain:
   $$\hat{x}_{t,n}^{(h)} = \text{softplus}\left( \text{softplus}^{-1}(\hat{x}_{t,n}^{\text{HAR},(h)}) + \alpha_n r_{b,n} \delta_{b,n} \right)$$
6. **Node-Balanced Objective:** Smooth-worst HAR-relative regret penalty ($\lambda=0.10, \tau=0.20$) to prevent buying aggregate performance at the expense of isolated market deterioration.

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
