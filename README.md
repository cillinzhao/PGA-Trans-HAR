# PGA-Trans-HAR

Forecasting code for **Forecasting Global Equity Volatility: Econometric
Anchoring and Cross-Market Attention**, by Xinlin Zhao and Haotian Qiao.

PGA-Trans-HAR learns a bounded neural adjustment around a frozen,
horizon-specific HAR forecast. Masked temporal attention and a rolling
VAR-GFEVD prior organize information across asynchronous markets. One
learned gate per destination market blends prior-guided and data-driven
spatial attention. Each market gate is fixed across dates and blocks after
training; graph priors and input representations vary over time.

## Release scope

This release provides the main forecasting model, six external baselines,
six controlled ablations, the paper input panel, and forecast comparison
with MSE, MAE, QLIKE, HAC/DM, Holm adjustment and MCS.

Economic evaluation implementations (VaR/ES and intraday utility), their
result files, figure-generation code and figures are not included. The
commands below reproduce the forecasting workflow covered by this release.

## Repository layout

```text
models/pga_trans_har.py           Main HAR-anchored attention model
baselines/baseline_v5_hn.py       Six forecasting baselines
ablation/pga_ablation.py         Controlled ablations
comparison_mse_retest.py         Forecast losses and statistical inference
data/df_union_sqrt.csv           Current paper panel
data/README.md                   Measurement, dates and source
data/panel_manifest.json         Data hash and sample boundaries
requirements.txt                Forecasting dependencies
LICENSE                         MIT license for code
NOTICE.md                       Third-party data scope
CITATION.cff                    Citation metadata
```

Existing model/baseline/ablation filenames are retained to fit the repository
layout. Their contents implement the current MSE experiment. Output model
labels retain `PGA-Trans-HAR-V5-MSE-Retest` for compatibility with the
archived forecasts.

## Installation

Use a fresh Python 3.12 environment and install a PyTorch build suitable
for your hardware. From the repository root:

```bash
python -m pip install -r requirements.txt
```

No Git command is required to run a downloaded repository ZIP. CUDA is
optional; training accepts `--device auto`, `--device cpu` or `--device cuda`.
The dependency file gives compatibility ranges, not a byte-exact lock of
the archived CUDA training environment.

## Data and forecast definition

The panel contains 3,531 union-calendar dates from 2013-03-05 through
2026-09-28, ordered SPX, GDAXI, FCHI, FTSE, OMXSPI, N225, KS11, HSI.
Values are daily Parkinson high-low volatility in percent; use the default
`--input-format vol_pct`. The 2,870 common active dates define a separate
evaluation restriction, not a replacement training sample.

The input filename is unchanged, but this panel replaces the earlier
2006-2022 repository dataset. Its hash and provenance are documented in
[data/README.md](data/README.md). Third-party market data are outside the
code MIT grant; see [NOTICE.md](NOTICE.md).

Horizons 1, 5 and 22 identify single future union-calendar target dates;
they are not cumulative h-day volatility. The 22-position input window
retains missing-market masks.

## Run the forecasting workflow

Run these commands from the repository root, in order:

```bash
python models/pga_trans_har.py
python baselines/baseline_v5_hn.py
python ablation/pga_ablation.py --suite controlled
python comparison_mse_retest.py
```

All scripts resolve their default input and output paths from their file
locations. Do not move only one script to another folder: the baseline and
ablation modules load the shared protocol from `models/pga_trans_har.py`.

Defaults use horizons 1/5/22 and neural seeds 1/2/3/4/5. The first 60 percent
of calendar positions form the training sample, the next 10 percent the
validation sample and the last 30 percent the test sample. Training ends
at exclusive index 2118 and validation at 2471. Epoch selection on the
60/10 split is followed by a fresh refit on the first 70 percent.

The six baselines are HAR, VHAR, HAR-KS, GNN-HAR, DCRNN-HAR and
Pure-ST-Transformer. The controlled ablations are no_prior, prior_only,
fixed_half, scalar_gate, no_temporal and mse_only. Optional legacy/expanded
ablations are supported by the script but are not the paper's controlled
six-variant comparison.

The full neural grid takes substantial time. A reduced-seed or one-horizon
diagnostic is a different experiment and does not reproduce the reported
five-seed ensemble. Use fresh output directories when changing settings.

## Protocol and outputs

| Directory | Contents |
| --- | --- |
| `results/main/` | Main-model configuration, forecasts and diagnostics |
| `results/baselines/` | Baseline configurations and forecasts |
| `results/ablation/` | Controlled-variant configurations and forecasts |
| `results/comparison/` | Forecast loss tables, seed summaries, HAC/DM and MCS |

The main model uses standardized MSE plus a HAR-relative smooth-worst
term. Neural baselines use node-balanced standardized MSE. QLIKE is an
out-of-sample variance-scale score using squared volatility targets and
predictions, with a volatility floor of 0.0001 percent.

DCRNN-HAR uses an origin-specific graph, autoregressive decoding and
intermediate-step training targets. Its reported setting
`--dcrnn-decoder-calendar observed` treats the observed activity mask as a
known trading schedule. `all_open` is a separate alternative setting.

The comparison script validates dataset/configuration identity, horizons,
seeds, origins, target dates and paired observed targets. It averages seed
forecasts before scoring and gives markets equal weight. Individual-seed
dispersion is reported separately. Common-day results restrict the same
trained models to dates with all eight markets active.

DM uses HAC errors and Holm adjustment. MCS uses the seven primary models,
alpha 0.1, 1,000 circular block-bootstrap replications and bootstrap seed
2026. `--skip-inference` generates loss tables without the paper's DM/MCS
results. `--run-family qlike` is a compatibility option for separately
supplied legacy forecasts; the training workflow above is the MSE experiment.

## Reference forecasting result

In the archived equal-market all-days evaluation, PGA-Trans-HAR has the
lowest QLIKE among the seven primary models at all three reported horizons.

| Horizon | HAR QLIKE | PGA-Trans-HAR QLIKE |
| --- | ---: | ---: |
| 1 | 0.366818 | 0.353356 |
| 5 | 0.504370 | 0.502187 |
| 22 | 0.624024 | 0.610207 |

These are reference values from archived forecasts. No full neural training
rerun was performed while preparing this repository-layout update.
Library versions and hardware can affect newly trained forecasts.

## Citation and license

Please cite the accompanying preprint. Author and title metadata are in
[CITATION.cff](CITATION.cff). Original code and documentation use the
[MIT license](LICENSE); the data scope is stated separately in [NOTICE.md](NOTICE.md).
