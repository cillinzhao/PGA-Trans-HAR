"""Forecast-origin-safe MSE-loss ablations for the fixed-gate PGA-Trans-HAR model.

Run, for example::

    python ablation/pga_ablation.py --suite controlled

The defaults resolve the data, final main-model code, and completed full-model
results from this project's known locations. No test-set metric is used to
select a variant, epoch, or hyperparameter. The completed full model is read
only as a locked reference; it is not silently retrained.

Primary comparisons change one specified design choice relative to the full
model. The optional ``legacy`` suite reproduces the *structural ladder* from
the earlier ablation script; its rows are not one-factor causal effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT / "models"))
import pga_trans_har as core  # noqa: E402

DEFAULT_DATA = PROJECT / "data/df_union_sqrt.csv"
DEFAULT_REFERENCE = PROJECT / "results" / "main"
DEFAULT_BASELINES = PROJECT / "results" / "baselines"
PRIMARY = (
    "no_prior", "prior_only", "fixed_half", "scalar_gate",
    "no_temporal", "mse_only",
)
EXTENDED = ("allow_inactive_sources", "mean_anchor")
LEGACY = ("legacy_temporal", "legacy_prior", "legacy_temporal_prior")
DESCRIPTIONS = {
    "no_prior": "g=1: data-driven spatial attention only; HAR, temporal attention and loss retained",
    "prior_only": "g=0: rolling prior only; HAR, temporal attention and loss retained",
    "fixed_half": "g=0.5 for every market; only learned market-specific allocation removed",
    "scalar_gate": "one learned gate shared by eight markets instead of eight learned gates",
    "no_temporal": "temporal-attention output set to zero; other block paths retained",
    "mse_only": "auxiliary HAR-relative smooth-worst loss removed; pure node-standardized panel MSE",
    "allow_inactive_sources": "closure mask retained as input/HAR feature but not applied to attention keys",
    "mean_anchor": "HAR anchor replaced by a frozen training-only market mean",
    "legacy_temporal": "old ladder: HAR plus temporal attention; spatial block absent",
    "legacy_prior": "old ladder: HAR plus rolling-prior aggregation; temporal block absent",
    "legacy_temporal_prior": "old ladder: HAR plus temporal attention and rolling-prior aggregation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ZeroTemporal(nn.Module):
    def forward(self, x, x_mask):
        return torch.zeros_like(x)


class UnmaskedTemporal(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, x_mask):
        return self.original(x, torch.ones_like(x_mask))


class UnmaskedSpatial(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x, x_mask, prior, gate_by_node, return_gate=False):
        return self.original(
            x, torch.ones_like(x_mask), prior, gate_by_node,
            return_gate=return_gate,
        )


class PriorGraphLayer(nn.Module):
    """The mask-aware prior aggregator from the original ablation ladder."""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, prior):
        batch, steps, nodes, _ = x.shape
        graph = prior[:, None].expand(batch, steps, nodes, nodes)
        graph = graph * x_mask[..., 0][:, :, None, :]
        total = graph.sum(dim=-1, keepdim=True)
        graph = graph / total.clamp_min(1e-12)
        # A union-calendar observation should contain at least one active
        # source. The fallback only guards degenerate external datasets.
        identity = torch.eye(nodes, device=x.device, dtype=x.dtype)[None, None]
        graph = torch.where(total > 1e-12, graph, identity)
        value = self.value(x)
        return self.output(self.dropout(torch.einsum("bsij,bsjh->bsih", graph, value)))


class LegacyBlock(nn.Module):
    """Original nested architecture, explicitly separate from primary tests."""

    def __init__(self, variant, hidden_dim, num_heads, dropout):
        super().__init__()
        self.temporal = (
            core.MultiHeadTemporalAttention(hidden_dim, num_heads, dropout)
            if variant != "legacy_prior" else None
        )
        self.graph = (
            PriorGraphLayer(hidden_dim, dropout)
            if variant != "legacy_temporal" else None
        )
        self.norm_temporal = nn.LayerNorm(hidden_dim) if self.temporal else None
        self.norm_graph = nn.LayerNorm(hidden_dim) if self.graph else None
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, prior, gate_by_node, return_gate=False):
        if self.temporal is not None:
            x = self.norm_temporal(x + self.dropout(self.temporal(x, x_mask)))
        if self.graph is not None:
            x = self.norm_graph(x + self.graph(x, x_mask, prior))
        x = self.norm_final(x + self.ffn(x))
        if not return_gate:
            return x, None
        gate_value = float("nan") if self.graph is None else 0.0
        return x, torch.full(x.shape[:3], gate_value, device=x.device, dtype=x.dtype)


class AblationModel(core.PGATransHARV5):
    def __init__(self, variant, mean_anchor, **kwargs):
        super().__init__(**kwargs)
        self.variant = variant
        self.register_buffer(
            "training_mean_anchor", torch.as_tensor(mean_anchor, dtype=torch.float32)
        )
        if variant == "scalar_gate":
            self.attention_gate_logits = nn.Parameter(torch.full((1,), -0.5))
        elif variant in {"no_prior", "prior_only", "fixed_half"}:
            logit = {"no_prior": 100.0, "prior_only": -100.0, "fixed_half": 0.0}[variant]
            self.attention_gate_logits = nn.Parameter(
                torch.full((self.num_nodes,), logit), requires_grad=False
            )
            if variant == "prior_only":
                # On union-calendar dates at least one source is active, so
                # Q/K scores cannot affect a g=0 prediction. Do not count
                # those dormant parameters as trainable capacity.
                for block in self.blocks:
                    for layer in (block.spatial.q, block.spatial.k):
                        for parameter in layer.parameters():
                            parameter.requires_grad_(False)
        elif variant == "no_temporal":
            for block in self.blocks:
                block.temporal = ZeroTemporal()
        elif variant == "allow_inactive_sources":
            for block in self.blocks:
                block.temporal = UnmaskedTemporal(block.temporal)
                block.spatial = UnmaskedSpatial(block.spatial)
        elif variant in LEGACY:
            cfg = kwargs
            self.attention_gate_logits.requires_grad_(False)
            self.blocks = nn.ModuleList([
                LegacyBlock(variant, cfg["hidden_dim"], cfg["num_heads"], cfg["dropout"])
                for _ in range(cfg["num_blocks"])
            ])

    def har_anchor(self, x, x_mask):
        if self.variant == "mean_anchor":
            return self.training_mean_anchor[None, :, None].expand(x.shape[0], -1, -1)
        return super().har_anchor(x, x_mask)

    def forward(self, x, x_mask, prior, return_aux=False):
        # Identical to the final-model forward pass except the one-scalar-gate
        # case expands a scalar rather than using N independent parameters.
        hidden = (
            self.input_projection(torch.cat([x, x_mask], dim=-1))
            + self.position_embedding + self.node_embedding
        )
        gate_by_node = torch.sigmoid(self.attention_gate_logits)
        if self.variant == "scalar_gate":
            gate_by_node = gate_by_node.expand(self.num_nodes)
        gates = []
        for block in self.blocks:
            hidden, gate = block(
                hidden, x_mask, prior, gate_by_node,
                return_gate=return_aux,
            )
            if return_aux:
                gates.append(gate)
        last_hidden = hidden[:, -1]
        active_rate = x_mask[..., 0].mean(dim=1)[..., None]
        residual_gate = torch.sigmoid(self.residual_gate(
            torch.cat([last_hidden, active_rate, x_mask[:, -1]], dim=-1)
        ))
        residual = torch.tanh(self.residual_head(last_hidden))
        alpha = self.max_residual_scale * torch.tanh(self.alpha_raw)[None]
        anchor = self.har_anchor(x, x_mask)
        latent = core.inverse_softplus(anchor) + alpha * residual_gate * residual
        prediction = torch.nn.functional.softplus(latent).clamp_min(core.EPS)
        if not return_aux:
            return prediction
        return prediction, {
            "attention_gate": torch.stack(gates, dim=0),
            "fixed_attention_gate_by_node": gate_by_node,
            "residual_gate": residual_gate,
            "alpha": alpha,
            "har_anchor": anchor,
            "residual": residual,
        }


def build_model(variant, values, end, har_coeff, config, seed, device):
    core.set_seed(seed)
    mean_anchor = np.nanmean(values[:end], axis=0)
    if not np.all(np.isfinite(mean_anchor)) or np.any(mean_anchor <= 0):
        raise ValueError("Each market needs a positive training-only mean anchor.")
    return AblationModel(
        variant=variant, mean_anchor=mean_anchor,
        num_nodes=values.shape[1], har_coefficients=har_coeff,
        seq_len=config.seq_len, hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks, num_heads=config.num_heads,
        dropout=config.dropout, max_residual_scale=config.max_residual_scale,
    ).to(device)


def verify_reference(reference, baselines, cfg, panel, horizons, seeds, train_end, val_end, data_hash):
    expected = json.loads((reference / "run_config.json").read_text(encoding="utf-8"))
    if expected.get("data_sha256") != data_hash:
        raise ValueError("Full-model results were produced from a different data file.")
    if expected["rows"] != len(panel.dates) or expected["nodes"] != len(panel.names):
        raise ValueError("Input panel dimensions differ from the trained full model.")
    if expected["train_end_exclusive"] != train_end or expected["validation_end_exclusive"] != val_end:
        raise ValueError("Input-panel split differs from the trained full model.")
    if expected["train_config"] != asdict(cfg):
        raise ValueError("Training configuration differs from the full-model run_config.json.")
    if not set(horizons).issubset(expected["horizons"]) or not set(seeds).issubset(expected["seeds"]):
        raise ValueError("Requested horizons/seeds are absent from full-model results.")
    baseline_cfg = json.loads((baselines / "run_config.json").read_text(encoding="utf-8"))
    if baseline_cfg.get("data_sha256") != data_hash:
        raise ValueError("Baseline results were produced from a different data file.")
    if baseline_cfg["rows"] != len(panel.dates) or tuple(baseline_cfg["names"]) != panel.names:
        raise ValueError("Saved HAR/Pure-ST baselines use another market panel.")
    baseline_predictions = pd.read_csv(baselines / "all_seed_predictions_ALL_HORIZONS.csv")
    for horizon in horizons:
        windows = core.make_windows(
            panel.values, val_end + horizon - 1,
            len(panel.dates), cfg.seq_len, horizon,
        )
        manifest, _ = core.date_manifests(windows, panel.dates, panel.names, horizon)
        original = pd.read_csv(reference / f"h{horizon}" / f"test_date_manifest_all_days_h{horizon}.csv")
        fields = ["horizon", "target_idx", "target_date", "origin_idx", "origin_date",
                  "node", "node_name", "active", "all_markets_active"]
        pd.testing.assert_frame_equal(
            manifest[fields].reset_index(drop=True), original[fields].reset_index(drop=True),
            check_dtype=False,
        )
        predictions = pd.read_csv(reference / f"h{horizon}" / f"all_seed_predictions_h{horizon}.csv")
        one_seed = predictions[predictions.seed == seeds[0]].reset_index(drop=True)
        expected_y = windows[2][..., 0].reshape(-1)
        expected_mask = windows[3][..., 0].reshape(-1) > 0.5
        if len(one_seed) != len(expected_y) or not np.array_equal(
            one_seed.active.to_numpy(dtype=bool), expected_mask
        ) or not np.allclose(
            one_seed.loc[expected_mask, "y_sqrt_rv_pct"].to_numpy(),
            expected_y[expected_mask], rtol=2e-6, atol=2e-6,
        ):
            raise ValueError("Input values do not match the saved full-model test targets.")
        for model_name, selected_seed in (("HAR", 0), ("Pure-ST-Transformer", seeds[0])):
            existing = baseline_predictions[
                (baseline_predictions.horizon == horizon) &
                (baseline_predictions.model == model_name) &
                (baseline_predictions.seed == selected_seed)
            ].reset_index(drop=True)
            if len(existing) != len(expected_y) or not np.array_equal(
                existing.active.to_numpy(dtype=bool), expected_mask
            ) or not np.allclose(
                existing.loc[expected_mask, "y_sqrt_rv_pct"],
                expected_y[expected_mask], rtol=2e-6, atol=2e-6,
            ):
                raise ValueError(f"Saved {model_name} baseline is not paired with this panel.")
    return expected


def selected_variants(args):
    if args.variants:
        variants = list(dict.fromkeys(args.variants))
    elif args.suite == "controlled":
        variants = list(PRIMARY)
    elif args.suite == "expanded":
        variants = list(PRIMARY + EXTENDED)
    elif args.suite == "legacy":
        variants = list(LEGACY)
    else:
        variants = list(PRIMARY + EXTENDED + LEGACY)
    unknown = set(variants) - set(DESCRIPTIONS)
    if unknown:
        raise ValueError(f"Unknown variants: {sorted(unknown)}")
    return variants


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--outdir", type=Path, default=PROJECT / "results" / "ablation")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--suite", choices=("controlled", "expanded", "legacy", "all"),
                        default="controlled")
    parser.add_argument("--variants", nargs="+", default=None, choices=tuple(DESCRIPTIONS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--resume", action="store_true", help="Skip only completed runs with matching experiment metadata.")
    parser.add_argument("--audit-only", action="store_true", help="Validate paths, sample, labels and split without training.")
    return parser.parse_args()


def aggregate_outputs(outdir, variants, horizons, seeds, reference, baselines):
    metrics, predictions, histories, inventory = [], [], [], []
    baseline_predictions = pd.read_csv(baselines / "all_seed_predictions_ALL_HORIZONS.csv")
    baseline_metrics = pd.read_csv(baselines / "all_seed_metrics_ALL_HORIZONS.csv")
    keep_predictions = (
        ((baseline_predictions.model == "HAR") & (baseline_predictions.seed == 0)) |
        ((baseline_predictions.model == "Pure-ST-Transformer") & baseline_predictions.seed.isin(seeds))
    ) & baseline_predictions.horizon.isin(horizons)
    keep_metrics = (
        ((baseline_metrics.model == "HAR") & (baseline_metrics.seed == 0)) |
        ((baseline_metrics.model == "Pure-ST-Transformer") & baseline_metrics.seed.isin(seeds))
    ) & baseline_metrics.horizon.isin(horizons)
    predictions.append(baseline_predictions.loc[keep_predictions].copy())
    metrics.append(baseline_metrics.loc[keep_metrics].copy())
    for horizon in horizons:
        hdir = outdir / f"h{horizon}"
        # Import the *existing*, never retuned main-model forecasts so paired
        # comparison uses exactly the same target dates and market activity.
        reference_predictions = pd.read_csv(reference / f"h{horizon}" / f"all_seed_predictions_h{horizon}.csv")
        reference_metrics = pd.read_csv(reference / f"h{horizon}" / f"all_seed_metrics_h{horizon}.csv")
        predictions.append(reference_predictions[reference_predictions.seed.isin(seeds)].copy())
        metrics.append(reference_metrics[reference_metrics.seed.isin(seeds)].copy())
        for variant in variants:
            for seed in seeds:
                run = hdir / variant / f"seed{seed}"
                if not (run / "done.json").exists():
                    continue
                metrics.append(pd.read_csv(run / "metrics.csv"))
                predictions.append(pd.read_csv(run / "predictions.csv"))
                history = pd.read_csv(run / "validation_history.csv")
                history.insert(0, "model", variant)
                history.insert(1, "horizon", horizon)
                history.insert(2, "seed", seed)
                histories.append(history)
                inventory.append(json.loads((run / "done.json").read_text(encoding="utf-8")))
    metric_frame = pd.concat(metrics, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame.to_csv(outdir / "all_seed_metrics_ALL_HORIZONS.csv", index=False)
    prediction_frame.to_csv(outdir / "all_seed_predictions_ALL_HORIZONS.csv", index=False)
    core.summarize_metrics(metric_frame).to_csv(outdir / "summary_mean_std_ALL_HORIZONS.csv", index=False)
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(outdir / "validation_history_ALL_HORIZONS.csv", index=False)
    pd.DataFrame(inventory).to_csv(outdir / "model_inventory.csv", index=False)
    ensembles, ensemble_metrics = [], []
    for (model_name, horizon), group in prediction_frame.groupby(["model", "horizon"]):
        ensemble, scores = core.ensemble_outputs(group, horizon)
        ensemble["model"] = model_name + "-SeedEnsemble"
        scores["model"] = model_name + "-SeedEnsemble"
        ensembles.append(ensemble)
        ensemble_metrics.append(scores)
    ensemble_frame = pd.concat(ensembles, ignore_index=True)
    ensemble_frame.to_csv(outdir / "ensemble_predictions_ALL_HORIZONS.csv", index=False)
    ensemble_score_frame = pd.concat(ensemble_metrics, ignore_index=True)
    ensemble_score_frame.to_csv(outdir / "ensemble_metrics_ALL_HORIZONS.csv", index=False)
    averages = ensemble_score_frame.groupby(["model", "horizon"], as_index=False)[
        ["QLIKE", "MSE", "MAE"]
    ].mean()
    har = averages[averages.model == "HAR-SeedEnsemble"].set_index("horizon")
    full = averages[averages.model == core.MODEL_NAME + "-SeedEnsemble"].set_index("horizon")
    for metric in ("QLIKE", "MSE", "MAE"):
        averages[f"{metric}_reduction_pct_vs_HAR"] = averages.apply(
            lambda row: 100.0 * (1.0 - row[metric] / har.loc[row.horizon, metric]), axis=1
        )
        averages[f"{metric}_minus_full"] = averages.apply(
            lambda row: row[metric] - full.loc[row.horizon, metric], axis=1
        )
    averages.to_csv(outdir / "cross_market_equal_weight_ensemble_summary.csv", index=False)
    paired_mse_tests(ensemble_frame, variants, horizons).to_csv(
        outdir / "paired_mse_hac_ALL_HORIZONS.csv", index=False
    )


def hac_mean_test(differential, lag):
    """Two-sided NW/DM-style test of mean(variant MSE - full MSE)=0."""
    x = np.asarray(differential, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), n
    lag = min(lag, n - 1)
    center = x - x.mean()
    long_run_var = float(np.dot(center, center) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(center[k:], center[:-k]) / n)
        long_run_var += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    if long_run_var <= 0 or not np.isfinite(long_run_var):
        return float(x.mean()), float("nan"), float("nan"), n
    se = math.sqrt(long_run_var / n)
    statistic = float(x.mean() / se)
    pvalue = math.erfc(abs(statistic) / math.sqrt(2.0))
    return float(x.mean()), statistic, pvalue, n


def paired_mse_tests(ensembles, variants, horizons):
    """Compare frozen seed ensembles on exactly paired test dates/markets.

    This is an exploratory, pre-specified ablation comparison. It does not
    infer structural causality or treat five random seeds as five test samples.
    Holm adjustment is within each horizon/panel/market family of variants.
    """
    full_name = core.MODEL_NAME + "-SeedEnsemble"
    rows = []
    keys = ["horizon", "target_idx", "node"]
    for h in horizons:
        full = ensembles[(ensembles.model == full_name) & (ensembles.horizon == h)].copy()
        if full.empty:
            raise ValueError(f"Missing locked full-model ensemble at h={h}.")
        full = full.sort_values(keys)
        n_nodes = int(full.node.nunique())
        common_dates = set(full.groupby("target_idx").active.sum().loc[lambda s: s == n_nodes].index)
        for variant in variants:
            name = "Ablation-" + variant + "-SeedEnsemble"
            other = ensembles[(ensembles.model == name) & (ensembles.horizon == h)]
            if other.empty:
                raise ValueError(f"Missing ensemble {name}, h={h}.")
            merged = full.merge(other, on=keys, suffixes=("_full", "_ablation"), validate="one_to_one")
            if len(merged) != len(full) or not np.array_equal(
                merged.active_full.to_numpy(), merged.active_ablation.to_numpy()
            ):
                raise ValueError("Ablation and full model do not have identical evaluation masks.")
            active = merged.active_full.to_numpy(dtype=bool)
            if not np.allclose(
                merged.loc[active, "y_sqrt_rv_pct_full"],
                merged.loc[active, "y_sqrt_rv_pct_ablation"],
                rtol=2e-6, atol=2e-6,
            ):
                raise ValueError("Ablation and full model have different targets.")
            merged = merged.loc[active].copy()
            y = merged.y_sqrt_rv_pct_full.to_numpy(dtype=float)
            loss_full = (y - merged.pred_sqrt_rv_pct_full.to_numpy(dtype=float)) ** 2
            loss_variant = (y - merged.pred_sqrt_rv_pct_ablation.to_numpy(dtype=float)) ** 2
            merged["difference"] = loss_variant - loss_full
            for subset in ("all_days", "common_days_subset"):
                chosen = merged if subset == "all_days" else merged[merged.target_idx.isin(common_dates)]
                if chosen.empty:
                    continue
                slices = [("equal_active_market_daily_mean", chosen.groupby("target_idx").difference.mean())]
                slices += [
                    (str(name), group.set_index("target_idx").difference.sort_index())
                    for name, group in chosen.groupby("node_name_full")
                ]
                for market, series in slices:
                    lag = max(10, h - 1)
                    mean_diff, statistic, pvalue, n = hac_mean_test(series.to_numpy(), lag)
                    rows.append({
                        "model": "Ablation-" + variant, "variant": variant,
                        "horizon": h, "panel": subset, "market": market,
                        "n_target_dates": n, "hac_lag": min(lag, max(n - 1, 0)),
                        "mean_mse_ablation_minus_full": mean_diff,
                        "dm_hac_t": statistic, "p_two_sided": pvalue,
                        "positive_favors_full": True,
                    })
    tests = pd.DataFrame(rows)
    tests["p_holm_within_horizon_panel_market"] = np.nan
    for _, group in tests.groupby(["horizon", "panel", "market"]):
        valid = group.p_two_sided.dropna().sort_values()
        m = len(valid)
        adjusted = np.maximum.accumulate(valid.to_numpy() * np.arange(m, 0, -1))
        tests.loc[valid.index, "p_holm_within_horizon_panel_market"] = np.minimum(adjusted, 1.0)
    return tests


def main():
    args = parse_args()
    variants = selected_variants(args)
    horizons = sorted(set(args.horizons))
    seeds = sorted(set(args.seeds))
    if any(h < 1 for h in horizons) or any(seed < 0 for seed in seeds):
        raise ValueError("Horizon must be positive and seeds nonnegative.")
    if not args.data.is_file():
        raise FileNotFoundError(args.data)
    if not (args.reference_dir / "run_config.json").is_file():
        raise FileNotFoundError(args.reference_dir / "run_config.json")
    for name in ("run_config.json", "all_seed_predictions_ALL_HORIZONS.csv", "all_seed_metrics_ALL_HORIZONS.csv"):
        if not (args.baseline_dir / name).is_file():
            raise FileNotFoundError(args.baseline_dir / name)
    reference_cfg = json.loads((args.reference_dir / "run_config.json").read_text(encoding="utf-8"))
    config = core.TrainConfig(**reference_cfg["train_config"])
    if reference_cfg["input_format"] != "vol_pct":
        raise ValueError("This script expects the existing percent-volatility proxy panel.")
    panel = core.load_panel(args.data, input_format="vol_pct")
    train_end, val_end = core.fixed_split(len(panel.dates), .60, .10)
    verify_reference(args.reference_dir, args.baseline_dir, config, panel,
                     horizons, seeds, train_end, val_end, sha256(args.data))
    metadata = {
        "data_path": str(args.data.resolve()), "data_sha256": sha256(args.data),
        "core_path": str(Path(core.__file__).resolve()),
        "core_sha256": sha256(Path(core.__file__)),
        "reference_path": str(args.reference_dir.resolve()),
        "reference_config_sha256": sha256(args.reference_dir / "run_config.json"),
        "reference_model": reference_cfg["model"], "variants": variants,
        "baseline_path": str(args.baseline_dir.resolve()),
        "baseline_config_sha256": sha256(args.baseline_dir / "run_config.json"),
        "descriptions": {name: DESCRIPTIONS[name] for name in variants},
        "horizons": horizons, "seeds": seeds,
        "train_config": asdict(config), "graph": reference_cfg["dynamic_graph"],
        "split": [train_end, val_end], "target_type": "daily Parkinson volatility proxy, percent",
        "primary_metric": "training-standardized MSE on volatility; report raw MSE, MAE and QLIKE",
        "inference_note": "Use paired date-level HAC/DM and block-bootstrap MCS after all runs; seeds are not independent test dates.",
    }
    if args.audit_only:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("AUDIT PASSED: dates, market names, observed labels, splits and locked training recipe match.")
        return
    outdir = args.outdir.resolve()
    if "qlike" in outdir.name.lower():
        raise ValueError("Refusing to write MSE results into a QLIKE-named output directory.")
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "experiment_config.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"Output exists; use --resume or a new --outdir: {outdir}")
        prior_metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_metadata != metadata:
            raise ValueError("Cannot resume: code/data/reference or variant list changed.")
    elif any(outdir.iterdir()):
        raise FileExistsError(f"Non-empty output directory lacks experiment_config.json: {outdir}")
    else:
        manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    device = core.choose_device(args.device)
    values, dates, names = panel.values, panel.dates, panel.names
    graph = reference_cfg["dynamic_graph"]
    for h in horizons:
        print(f"Building aligned windows and origin-admissible graph: h={h}", flush=True)
        train_w = core.make_windows(values, config.seq_len, train_end, config.seq_len, h)
        val_w = core.make_windows(values, train_end + h - 1, val_end, config.seq_len, h)
        refit_w = core.make_windows(values, config.seq_len, val_end, config.seq_len, h)
        test_w = core.make_windows(values, val_end + h - 1, len(values), config.seq_len, h)
        if val_w[4][0] - h != train_end - 1 or test_w[4][0] - h != val_end - 1:
            raise AssertionError("Forecast-origin split alignment failed.")
        all_indices = np.unique(np.concatenate([w[4] for w in (train_w, val_w, refit_w, test_w)]))
        prior, diagnostics = core.build_dynamic_priors(
            values, all_indices, seq_len=config.seq_len,
            graph_window=graph["window"], graph_update_every=graph["update_every"],
            var_lag=graph["var_lag"], ridge=graph["ridge"],
            dy_horizon=graph["dy_horizon"], forecast_horizon=h,
        )
        prior_lookup = {int(idx): prior[i] for i, idx in enumerate(all_indices)}
        def dataset(windows):
            return core.ForecastDataset(
                windows,
                np.asarray([prior_lookup[int(idx)] for idx in windows[4]], dtype=np.float32),
            )
        train_ds, val_ds = dataset(train_w), dataset(val_w)
        refit_ds, test_ds = dataset(refit_w), dataset(test_w)
        val_loader = core.make_loader(val_ds, config.batch_size, False)
        test_loader = core.make_loader(test_ds, config.batch_size, False)
        har_dev = core.fit_har_anchor(values, train_end, config.seq_len, h)
        har_refit = core.fit_har_anchor(values, val_end, config.seq_len, h)
        reference_dev = core.training_statistics(values, train_end, config.seq_len, har_dev, h)
        reference_refit = core.training_statistics(values, val_end, config.seq_len, har_refit, h)
        hdir = outdir / f"h{h}"
        hdir.mkdir(exist_ok=True)
        diagnostics.to_csv(hdir / "graph_diagnostics.csv", index=False)
        for variant in variants:
            run_config = replace(config, robust_lambda=0.0) if variant == "mse_only" else config
            for seed in seeds:
                run = hdir / variant / f"seed{seed}"
                if (run / "done.json").exists():
                    print(f"Skipping complete: {variant} h={h} seed={seed}", flush=True)
                    continue
                run.mkdir(parents=True, exist_ok=True)
                print(f"Training {variant} h={h} seed={seed}", flush=True)
                dev_model = build_model(variant, values, train_end, har_dev, run_config, seed, device)
                dev_objective = core.RobustRelativeMSE(
                    *reference_dev, run_config.robust_lambda, run_config.robust_temperature,
                ).to(device)
                train_loader = core.make_loader(train_ds, run_config.batch_size, True)
                _, best_epoch, history = core.select_epoch_on_validation(
                    dev_model, train_loader, val_loader, dev_objective,
                    reference_dev, run_config, device,
                )
                del dev_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                refit_model = build_model(variant, values, val_end, har_refit, run_config, seed, device)
                refit_objective = core.RobustRelativeMSE(
                    *reference_refit, run_config.robust_lambda, run_config.robust_temperature,
                ).to(device)
                refit_loader = core.make_loader(refit_ds, run_config.batch_size, True)
                refit_model, final_loss = core.refit_fixed_epochs(
                    refit_model, refit_loader, refit_objective,
                    best_epoch, run_config, device,
                )
                arrays = core.collect_predictions(refit_model, test_loader, device, return_aux=True)
                model_name = "Ablation-" + variant
                metric = core.metric_table(
                    arrays["target"], arrays["prediction"], arrays["mask"], names, seed, h,
                )
                metric["model"] = model_name
                metric["selected_epoch"] = best_epoch
                metric["final_train_objective"] = final_loss
                predictions = core.prediction_frame(arrays, dates, names, seed, h)
                predictions["model"] = model_name
                history["selected_epoch"] = best_epoch
                metric.to_csv(run / "metrics.csv", index=False)
                predictions.to_csv(run / "predictions.csv", index=False)
                history.to_csv(run / "validation_history.csv", index=False)
                gate = arrays["attention_gate"][-1, :, -1, :]
                pd.DataFrame({
                    "target_idx": np.repeat(arrays["target_idx"], len(names)),
                    "node_name": np.tile(names, len(arrays["target_idx"])),
                    "attention_gate_g": gate.reshape(-1),
                    "residual_gate_r": arrays["residual_gate"].reshape(-1),
                    "har_or_mean_anchor": arrays["har_anchor"].reshape(-1),
                    "alpha": np.tile(
                        (run_config.max_residual_scale * torch.tanh(refit_model.alpha_raw.detach()))
                        .cpu().numpy().reshape(-1),
                        len(arrays["target_idx"]),
                    ),
                }).to_csv(run / "internal_diagnostics.csv", index=False)
                inventory = {
                    "model": model_name, "variant": variant, "horizon": h, "seed": seed,
                    "selected_epoch": best_epoch, "final_train_objective": final_loss,
                    "trainable_parameters": sum(p.numel() for p in refit_model.parameters() if p.requires_grad),
                    "total_parameters": sum(p.numel() for p in refit_model.parameters()),
                    "robust_lambda": run_config.robust_lambda,
                    "completed": True,
                }
                (run / "done.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
                print(f"Finished {variant} h={h} seed={seed}; epoch={best_epoch}", flush=True)
                del refit_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    missing = [
        f"{variant}/h{h}/seed{seed}"
        for h in horizons for variant in variants for seed in seeds
        if not (outdir / f"h{h}" / variant / f"seed{seed}" / "done.json").is_file()
    ]
    if missing:
        raise RuntimeError(f"Incomplete experiment; combined tables withheld: {missing[:10]}")
    aggregate_outputs(outdir, variants, horizons, seeds, args.reference_dir, args.baseline_dir)
    print(f"Ablation outputs: {outdir}")


if __name__ == "__main__":
    main()
