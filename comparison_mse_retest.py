"""Strict comparison of the MSE-retest full model, baselines and ablations.

Defaults locate main, baselines and ablation in the sibling results directory.
MSE is the primary evaluation metric; MAE and variance QLIKE are also reported.
All reported losses
are scored on the seed-MEAN volatility forecast, not a mean of seed losses.

Examples::

    python comparison_mse_retest.py
    python comparison_mse_retest.py --without-ablation --skip-inference \
        --outdir results/comparison_diagnostic

No model is trained by this script. Existing output directories are protected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_RESULTS = HERE / "results"
DEFAULT_ABLATION = DEFAULT_RESULTS / "ablation"
DEFAULT_OUT = DEFAULT_RESULTS / "comparison"
MAIN = "PGA-Trans-HAR-V5-MSE-Retest"
BASELINES = ("HAR", "VHAR", "HAR-KS", "GNN-HAR", "DCRNN-HAR", "Pure-ST-Transformer")
MAIN_POOL = BASELINES + (MAIN,)
METRICS = ("MSE", "MAE", "QLIKE")
FLOOR = 1e-4
REQUIRED = (
    "model", "horizon", "seed", "target_idx", "target_date", "origin_idx",
    "origin_date", "node", "node_name", "y_sqrt_rv_pct", "pred_sqrt_rv_pct", "active",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def options():
    global MAIN, MAIN_POOL, METRICS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-family", choices=("mse", "qlike"), default="mse",
                   help="MSE retest by default; qlike re-scores the saved QLIKE runs under the same inference code.")
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--main-dir", type=Path, default=None,
                   help="Override the main result directory; otherwise results-root/main.")
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="Override the baseline result directory; otherwise results-root/baselines.")
    p.add_argument("--ablation-dir", type=Path, default=None)
    p.add_argument("--panel-csv", type=Path,
                   default=HERE / "data/df_union_sqrt.csv")
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--without-ablation", action="store_true",
                   help="Explicit seven-model diagnostic only; main comparison requires ablations.")
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--mcs-alpha", type=float, default=.10)
    p.add_argument("--mcs-bootstrap", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--mcs-include-ablations", action="store_true",
                   help="Exploratory MCS including ablations; primary MCS uses only seven main models.")
    p.add_argument("--skip-inference", action="store_true", help="Only build alignment and loss tables.")
    a = p.parse_args()
    is_mse = a.run_family == "mse"
    MAIN = "PGA-Trans-HAR-V5-MSE-Retest" if is_mse else "PGA-Trans-HAR-V5-QLIKE"
    MAIN_POOL = BASELINES + (MAIN,)
    METRICS = ("MSE", "MAE", "QLIKE") if is_mse else ("QLIKE", "MSE", "MAE")
    a.results_root = a.results_root or DEFAULT_RESULTS
    a.main_dir = a.main_dir or a.results_root / ("main" if is_mse else "pga_results_qlike")
    a.baseline_dir = a.baseline_dir or a.results_root / ("baselines" if is_mse else "baseline_results_qlike")
    a.ablation_dir = a.ablation_dir or a.results_root / ("ablation" if is_mse else "qlike_ablation_results")
    a.outdir = a.outdir or (DEFAULT_OUT if is_mse else DEFAULT_RESULTS / "comparison_qlike_recheck")
    a.horizons = sorted(set(a.horizons))
    a.seeds = sorted(set(a.seeds))
    if not a.horizons or min(a.horizons) < 1 or not a.seeds:
        p.error("Horizons and seeds must be nonempty, and horizons positive.")
    if not 0 < a.mcs_alpha < 1 or a.mcs_bootstrap < 100:
        p.error("MCS needs 0<alpha<1 and at least 100 bootstrap replications.")
    return a


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_predictions(path: Path, horizons: list[int], allowed_models=None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, low_memory=False)
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    frame = frame[list(REQUIRED)]
    frame = frame[frame.horizon.isin(horizons)]
    if allowed_models is not None:
        frame = frame[frame.model.isin(allowed_models)]
    if frame.empty:
        raise ValueError(f"No selected predictions in {path}")
    return frame.copy()


def load_sources(args):
    pga_dir = args.main_dir
    baseline_dir = args.baseline_dir
    pga_cfg = read_json(pga_dir / "run_config.json")
    base_cfg = read_json(baseline_dir / "run_config.json")
    if pga_cfg.get("model") != MAIN:
        raise ValueError(f"Expected {MAIN}; result files do not match --run-family.")
    data_hash = sha256(args.panel_csv)
    for cfg in (pga_cfg, base_cfg):
        if (args.run_family == "mse" or cfg.get("data_sha256")) and cfg.get("data_sha256") != data_hash:
            raise ValueError("Main/baseline data hashes do not match the evaluation panel.")
    loss_name = "MSE" if args.run_family == "mse" else "QLIKE"
    if loss_name not in pga_cfg.get("training_objective", "") or loss_name not in base_cfg.get("neural_primary_loss", ""):
        raise ValueError(f"Expected {loss_name}-trained main and neural baselines.")
    for field in ("rows", "nodes", "train_end_exclusive", "validation_end_exclusive",
                  "test_first_target_date_by_horizon", "split_protocol"):
        if pga_cfg.get(field) != base_cfg.get(field):
            raise ValueError(f"Full model and baselines disagree on {field}")
    if pga_cfg.get("input_format") != "vol_pct" or base_cfg.get("input_format") != "vol_pct":
        raise ValueError("Expected same percent-volatility input scale in both runs.")
    if not set(args.horizons).issubset(pga_cfg["horizons"]) or not set(args.horizons).issubset(base_cfg["horizons"]):
        raise ValueError("A requested horizon is absent from saved model configurations.")
    if not set(args.seeds).issubset(pga_cfg["seeds"]) or not set(args.seeds).issubset(base_cfg["seeds"]):
        raise ValueError("A requested neural seed is absent from saved model configurations.")
    panel = pd.read_csv(args.panel_csv, index_col=0)
    panel.index = pd.to_datetime(panel.index, errors="raise")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing or tuple(panel.columns) != tuple(base_cfg["names"]):
        raise ValueError("Panel dates/market order disagree with saved baseline configuration.")
    if len(panel) != pga_cfg["rows"]:
        raise ValueError("Panel length disagrees with saved run configuration.")
    panel_values = panel.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    frames = [
        read_predictions(
            baseline_dir / "all_seed_predictions_ALL_HORIZONS.csv", args.horizons, BASELINES
        )
    ]
    for h in args.horizons:
        frames.append(read_predictions(
            pga_dir / f"h{h}/all_seed_predictions_h{h}.csv", [h], [MAIN]
        ))
    variants = []
    ablation_meta = None
    if not args.without_ablation:
        ablation_meta = read_json(args.ablation_dir / "experiment_config.json")
        variants = list(ablation_meta["variants"])
        if not variants:
            raise ValueError("Ablation manifest lists no variants.")
        if not set(args.horizons).issubset(ablation_meta["horizons"]) or not set(args.seeds).issubset(ablation_meta["seeds"]):
            raise ValueError("Requested horizons/seeds were not trained in ablation runs.")
        if ablation_meta["data_sha256"] != sha256(args.panel_csv):
            raise ValueError("Ablations used a different volatility panel.")
        if ablation_meta["reference_config_sha256"] != sha256(pga_dir / "run_config.json"):
            raise ValueError("Ablations were not fitted against this full-model recipe.")
        if ablation_meta.get("baseline_config_sha256") != sha256(baseline_dir / "run_config.json"):
            raise ValueError("Ablations reference a different baseline run.")
        if ablation_meta["train_config"] != pga_cfg["train_config"] or ablation_meta["graph"] != pga_cfg["dynamic_graph"]:
            raise ValueError("Ablation and full-model training/graph settings differ.")
        for h in args.horizons:
            for variant in variants:
                for seed in args.seeds:
                    directory = args.ablation_dir / f"h{h}" / variant / f"seed{seed}"
                    status = read_json(directory / "done.json")
                    if not status.get("completed") or status["variant"] != variant or status["horizon"] != h or status["seed"] != seed:
                        raise ValueError(f"Incomplete or mismatched ablation trial: {directory}")
                    frame = read_predictions(directory / "predictions.csv", [h], ["Ablation-" + variant])
                    frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    for column in ("horizon", "seed", "target_idx", "origin_idx", "node", "active",
                   "y_sqrt_rv_pct", "pred_sqrt_rv_pct"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    for column in ("target_date", "origin_date"):
        raw[column] = pd.to_datetime(raw[column], errors="raise").dt.normalize()
    if raw[["horizon", "seed", "target_idx", "origin_idx", "node", "active"]].isna().any().any():
        raise ValueError("Prediction keys or activity indicators contain NaN.")
    if (raw.target_idx - raw.origin_idx != raw.horizon).any():
        raise ValueError("At least one forecast has inconsistent horizon/origin indices.")
    raw = raw[raw.seed.isin(args.seeds) | ((raw.model.isin(("HAR", "VHAR", "HAR-KS"))) & (raw.seed == 0))]
    raw = raw.sort_values(["model", "horizon", "seed", "target_idx", "node"]).reset_index(drop=True)
    if raw.duplicated(["model", "horizon", "seed", "target_idx", "node"]).any():
        raise ValueError("Duplicate model/horizon/seed/date/market forecasts.")
    expected_models = list(MAIN_POOL) + ["Ablation-" + v for v in variants]
    if set(raw.model) != set(expected_models):
        raise ValueError(f"Missing or unexpected models: expected {expected_models}; got {sorted(set(raw.model))}")
    # One complete target-market grid and mask is required for every model and seed.
    first_seed = args.seeds[0]
    canonical = raw[(raw.model == MAIN) & (raw.seed == first_seed)].copy()
    canonical = canonical[["horizon", "target_idx", "node", "target_date", "origin_idx",
                           "origin_date", "node_name", "active", "y_sqrt_rv_pct"]]
    canonical = canonical.rename(columns={"y_sqrt_rv_pct": "y_reference"})
    target_idx = canonical.target_idx.to_numpy(dtype=int)
    origin_idx = canonical.origin_idx.to_numpy(dtype=int)
    node_idx = canonical.node.to_numpy(dtype=int)
    if (np.any(target_idx < 0) or np.any(target_idx >= len(panel)) or
            np.any(origin_idx < 0) or np.any(node_idx < 0) or np.any(node_idx >= panel.shape[1])):
        raise ValueError("Prediction indices are outside the evaluation panel.")
    if (not np.array_equal(canonical.target_date.to_numpy(), panel.index.to_numpy()[target_idx]) or
            not np.array_equal(canonical.origin_date.to_numpy(), panel.index.to_numpy()[origin_idx])):
        raise ValueError("Prediction dates do not match the panel at target/origin indices.")
    panel_targets = panel_values[target_idx, node_idx]
    observed = np.isfinite(panel_targets) & (panel_targets > 0)
    if not np.array_equal(canonical.active.to_numpy() == 1, observed):
        raise ValueError("Full-model target mask does not match the evaluation panel.")
    if not np.allclose(canonical.y_reference.to_numpy()[observed], panel_targets[observed], rtol=2e-6, atol=2e-6):
        raise ValueError("Full-model targets do not match the evaluation panel.")
    keys = ["horizon", "target_idx", "node"]
    merged = raw.merge(canonical, on=keys, how="left", validate="many_to_one", suffixes=("", "_ref"), indicator=True)
    if len(merged) != len(raw) or (merged._merge != "both").any():
        raise ValueError("A model has a target date/market absent from the full-model sample.")
    for field in ("target_date", "origin_idx", "origin_date", "node_name"):
        if not merged[field].equals(merged[field + "_ref"]):
            raise ValueError(f"A model and full PGA disagree on {field}.")
    if ((merged.active == 1) & (merged.active_ref != 1)).any():
        raise ValueError("A model declares a forecast active where the target market was closed.")
    active = merged.active.eq(1)
    if not np.isfinite(merged.loc[active, "pred_sqrt_rv_pct"]).all() or (merged.loc[active, "pred_sqrt_rv_pct"] <= 0).any():
        raise ValueError("Active forecasts must be finite and positive.")
    if not np.allclose(merged.loc[active, "y_sqrt_rv_pct"], merged.loc[active, "y_reference"],
                       rtol=2e-6, atol=2e-6):
        raise ValueError("Model target values disagree with full PGA.")
    expected_grid = canonical.groupby("horizon").size().to_dict()
    counts = raw.groupby(["model", "horizon", "seed"]).size()
    for (model, h, seed), count in counts.items():
        if count != expected_grid[h]:
            raise ValueError(f"Incomplete forecast grid: {model}, h={h}, seed={seed}: {count}/{expected_grid[h]}")
    for model in expected_models:
        for h in args.horizons:
            observed_seeds = set(raw[(raw.model == model) & (raw.horizon == h)].seed)
            wanted = {0} if model in ("HAR", "VHAR", "HAR-KS") else set(args.seeds)
            if observed_seeds != wanted:
                raise ValueError(f"Missing seed(s): {model}, h={h}, got {observed_seeds}, expected {wanted}")
    return raw, canonical, expected_models, variants, pga_cfg, base_cfg, ablation_meta


def build_ensemble(raw: pd.DataFrame, canonical: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "horizon", "target_idx", "node"]
    ensemble = raw.groupby(keys, as_index=False).agg(
        pred_sqrt_rv_pct=("pred_sqrt_rv_pct", "mean"),
        pred_seed_std=("pred_sqrt_rv_pct", "std"),
        n_seeds=("seed", "nunique"),
        active=("active", "min"),
        max_active=("active", "max"),
    )
    ensemble = ensemble.merge(
        canonical, on=["horizon", "target_idx", "node"],
        validate="many_to_one", suffixes=("", "_target")
    )
    ensemble = ensemble.rename(columns={"y_reference": "y_sqrt_rv_pct", "active_target": "target_active"})
    if (ensemble.active != ensemble.max_active).any():
        raise ValueError("Seeds of one model disagree about activity on a market-date.")
    if ((ensemble.active == 1) & (ensemble.target_active != 1)).any():
        raise ValueError("Active forecast on a closed target market-day.")
    ensemble["pred_seed_std"] = ensemble["pred_seed_std"].fillna(0.0)
    return ensemble.drop(columns="max_active").sort_values(["horizon", "model", "target_idx", "node"])


def align_active_losses(ensemble: pd.DataFrame, models: list[str]):
    scored = score_losses(ensemble)
    keys = ["horizon", "target_idx", "node"]
    counts = scored.groupby(keys).model.nunique()
    common_keys = counts[counts == len(models)].index
    selected = pd.MultiIndex.from_frame(scored[keys]).isin(common_keys)
    aligned = scored.loc[selected].copy()
    if aligned.empty or aligned.groupby(keys).model.nunique().ne(len(models)).any():
        raise ValueError("No complete paired market-date loss matrix remains.")
    audit = scored.groupby(["model", "horizon"], as_index=False).agg(
        active_before_alignment=("active", "size")
    )
    retained = aligned.groupby(["model", "horizon"], as_index=False).agg(
        active_after_alignment=("active", "size")
    )
    audit = audit.merge(retained, on=["model", "horizon"], validate="one_to_one")
    audit["dropped_for_common_model_sample"] = (
        audit.active_before_alignment - audit.active_after_alignment
    )
    return aligned, audit


def score_losses(panel: pd.DataFrame) -> pd.DataFrame:
    work = panel[panel.active.eq(1)].copy()
    y = work.y_sqrt_rv_pct.to_numpy(dtype=float)
    pred = work.pred_sqrt_rv_pct.to_numpy(dtype=float)
    log_ratio = 2.0 * (np.log(np.maximum(y, FLOOR)) - np.log(np.maximum(pred, FLOOR)))
    work["QLIKE"] = np.expm1(log_ratio) - log_ratio
    work["MSE"] = (pred - y) ** 2
    work["MAE"] = np.abs(pred - y)
    if not np.isfinite(work[list(METRICS)]).all().all():
        raise ValueError("At least one active forecast has nonfinite QLIKE/MSE/MAE.")
    return work


def summarize(losses: pd.DataFrame, panel_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = losses.groupby(["model", "horizon", "node", "node_name"], as_index=False)
    market = group.agg(n=("QLIKE", "size"), n_seeds=("n_seeds", "min"),
                       QLIKE=("QLIKE", "mean"), MSE=("MSE", "mean"), MAE=("MAE", "mean"))
    market.insert(0, "panel", panel_name)
    cross = market.groupby(["model", "horizon"], as_index=False).agg(
        n_markets=("node", "nunique"), QLIKE=("QLIKE", "mean"),
        MSE=("MSE", "mean"), MAE=("MAE", "mean"),
    )
    cross.insert(0, "panel", panel_name)
    if (cross.n_markets != 8).any():
        raise ValueError("All eight markets must have scored observations.")
    return market, cross


def seed_summaries(raw: pd.DataFrame, aligned: pd.DataFrame, common: pd.DataFrame):
    """Score each seed on the same paired observations as the ensemble tables."""
    keys = ["horizon", "target_idx", "node"]
    scored = []
    for panel_name, sample in (("all_days", aligned), ("common_days", common)):
        valid_keys = sample[keys].drop_duplicates()
        seed_rows = raw.merge(valid_keys, on=keys, how="inner", validate="many_to_one")
        if seed_rows.active.ne(1).any():
            raise ValueError("An aligned market-day is inactive for one seed.")
        seed_loss = score_losses(seed_rows)
        metrics = seed_loss.groupby(
            ["model", "horizon", "seed", "node", "node_name"], as_index=False
        ).agg(n=("QLIKE", "size"), QLIKE=("QLIKE", "mean"),
              MSE=("MSE", "mean"), MAE=("MAE", "mean"))
        metrics.insert(0, "panel", panel_name)
        scored.append(metrics)
    per_seed = pd.concat(scored, ignore_index=True)
    market_summary = per_seed.groupby(
        ["panel", "model", "horizon", "node", "node_name"], as_index=False
    ).agg(n_seeds=("seed", "nunique"), n=("n", "min"),
          QLIKE_mean=("QLIKE", "mean"), QLIKE_std=("QLIKE", "std"),
          MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
          MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"))
    for field in ("QLIKE_std", "MSE_std", "MAE_std"):
        market_summary[field] = market_summary[field].fillna(0.0)
    cross_seed = per_seed.groupby(["panel", "model", "horizon", "seed"], as_index=False)[
        list(METRICS)
    ].mean()
    cross_summary = cross_seed.groupby(["panel", "model", "horizon"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        QLIKE_mean=("QLIKE", "mean"), QLIKE_std=("QLIKE", "std"),
        MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
    )
    for field in ("QLIKE_std", "MSE_std", "MAE_std"):
        cross_summary[field] = cross_summary[field].fillna(0.0)
    return per_seed, market_summary, cross_summary


def hac_test(d: np.ndarray, h: int) -> dict:
    x = np.asarray(d, dtype=np.float64)
    n = len(x)
    lag = min(n - 1, max(h - 1, int(math.floor(4 * (n / 100) ** (2 / 9)))))
    center = x - x.mean()
    long_run = float(np.dot(center, center) / n)
    for k in range(1, lag + 1):
        long_run += 2 * (1 - k / (lag + 1)) * float(np.dot(center[k:], center[:-k]) / n)
    if n < 20 or long_run <= 0:
        stat, p = float("nan"), float("nan")
    else:
        stat = float(x.mean() / math.sqrt(long_run / n))
        p = math.erfc(abs(stat) / math.sqrt(2))
    return {"n_dates": n, "hac_lag": lag, "mean_competitor_minus_pga": float(x.mean()),
            "dm_hac_stat": stat, "p_two_sided": p}


def dm_tables(losses: pd.DataFrame, main_pool: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for (h, node, name), group in losses.groupby(["horizon", "node", "node_name"]):
        for metric in METRICS:
            matrix = group.pivot(index="target_idx", columns="model", values=metric)
            if matrix.isna().any().any() or MAIN not in matrix:
                raise ValueError("Paired loss matrix is incomplete.")
            for competitor in matrix.columns.difference([MAIN]):
                pool = "primary" if competitor in main_pool else "ablation"
                result = hac_test((matrix[competitor] - matrix[MAIN]).to_numpy(), int(h))
                rows.append({"horizon": int(h), "node": int(node), "node_name": name,
                             "metric": metric, "reference_model": MAIN,
                             "competitor_model": competitor, "competitor_pool": pool, **result})
    out = pd.DataFrame(rows)
    out["p_holm_within_pool"] = np.nan
    for _, group in out.groupby(["horizon", "node", "metric", "competitor_pool"]):
        valid = group.p_two_sided.dropna().sort_values()
        m = len(valid)
        adjusted = np.maximum.accumulate(valid.to_numpy() * np.arange(m, 0, -1))
        out.loc[valid.index, "p_holm_within_pool"] = np.minimum(adjusted, 1.0)
    return out


def circular_blocks(n: int, block: int, draws: int, rng: np.random.Generator):
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    offsets = np.arange(block)
    return ((starts[:, :, None] + offsets) % n).reshape(draws, -1)[:, :n]


def mcs_range(matrix: pd.DataFrame, h: int, alpha: float, draws: int, seed: int):
    if matrix.isna().any().any() or len(matrix) < 20 or matrix.shape[1] < 2:
        raise ValueError("MCS needs complete paired losses on at least 20 dates and two models.")
    n = len(matrix)
    block = max(2, h, int(round(n ** (1 / 3))))
    indices = circular_blocks(n, block, draws, np.random.default_rng(seed))
    remaining = list(map(str, matrix.columns))
    history = []
    while len(remaining) > 1:
        x = matrix[remaining].to_numpy(dtype=float)
        average = x.mean(axis=0)
        bootstrap = x[indices].mean(axis=1)
        difference = average[:, None] - average[None, :]
        boot_difference = bootstrap[:, :, None] - bootstrap[:, None, :] - difference
        se = np.sqrt(np.mean(boot_difference ** 2, axis=0))
        upper = np.triu_indices(len(remaining), 1)
        scale = np.maximum(se[upper], 1e-12)
        observed_t = np.abs(difference[upper] / scale)
        bootstrap_t = np.abs(boot_difference[:, upper[0], upper[1]] / scale[None])
        observed_stat = float(observed_t.max())
        bootstrap_stat = bootstrap_t.max(axis=1)
        p = float(np.mean(bootstrap_stat >= observed_stat))
        if p >= alpha:
            history.append({"step": len(history), "p_value": p, "eliminated": "",
                            "remaining_models": "|".join(remaining), "block_length": block})
            break
        # The range statistic uses pairwise studentized loss differences.
        # Its matching elimination rule removes i maximizing max_j t_ij.
        standardized = difference / np.maximum(se, 1e-12)
        np.fill_diagonal(standardized, -np.inf)
        eliminated = remaining.pop(int(np.argmax(standardized.max(axis=1))))
        history.append({"step": len(history), "p_value": p, "eliminated": eliminated,
                        "remaining_models": "|".join(remaining), "block_length": block})
    return remaining, history


def mcs_tables(losses: pd.DataFrame, panel_name: str, pool: list[str], args):
    sets, histories = [], []
    subset = losses[losses.model.isin(pool)]
    for (h, node, name), group in subset.groupby(["horizon", "node", "node_name"]):
        for metric in METRICS:
            matrix = group.pivot(index="target_idx", columns="model", values=metric)
            if set(matrix.columns) != set(pool):
                raise ValueError(f"MCS model pool incomplete at h={h}, {name}, {metric}.")
            included, trace = mcs_range(
                matrix, int(h), args.mcs_alpha, args.mcs_bootstrap,
                args.bootstrap_seed + (0 if panel_name == "all_days" else 100000)
                + int(h) * 1000 + int(node) * 10 + {"QLIKE": 0, "MSE": 1, "MAE": 2}[metric],
            )
            sets.append({"panel": panel_name, "horizon": int(h), "node": int(node),
                         "node_name": name, "metric": metric, "alpha": args.mcs_alpha,
                         "n_dates": len(matrix), "n_models": len(pool),
                         "included_models": "|".join(included), "n_included": len(included)})
            histories.extend({"panel": panel_name, "horizon": int(h), "node": int(node),
                              "node_name": name, "metric": metric, **row} for row in trace)
    return pd.DataFrame(sets), pd.DataFrame(histories)


def ablation_gains(market: pd.DataFrame, variants: list[str]):
    full = market[market.model == MAIN].set_index(["panel", "horizon", "node"])
    rows = []
    for variant in variants:
        name = "Ablation-" + variant
        comparator = market[market.model == name].set_index(["panel", "horizon", "node"])
        if len(comparator) != len(full) or not comparator.index.equals(full.index):
            raise ValueError(f"Ablation {name} is missing aligned market cells.")
        for key in full.index:
            for metric in METRICS:
                full_value = float(full.loc[key, metric])
                base_value = float(comparator.loc[key, metric])
                rows.append({"panel": key[0], "horizon": key[1], "node": key[2],
                             "node_name": full.loc[key, "node_name"], "variant": variant,
                             "metric": metric, "ablation_loss": base_value,
                             "full_loss": full_value,
                             "loss_reduction_full_vs_ablation_pct":
                             100 * (base_value - full_value) / base_value,
                             "positive_favors_full": bool(base_value > full_value)})
    return pd.DataFrame(rows)


def baseline_dominance(market: pd.DataFrame):
    full = market[market.model == MAIN].set_index(["panel", "horizon", "node"])
    rows = []
    for baseline in BASELINES:
        comparator = market[market.model == baseline].set_index(["panel", "horizon", "node"])
        if not comparator.index.equals(full.index):
            raise ValueError(f"Baseline {baseline} has missing aligned market cells.")
        for key in full.index:
            for metric in METRICS:
                base = float(comparator.loc[key, metric])
                pga = float(full.loc[key, metric])
                rows.append({"panel": key[0], "horizon": key[1], "node": key[2],
                             "node_name": full.loc[key, "node_name"],
                             "baseline": baseline, "metric": metric,
                             "baseline_loss": base, "pga_loss": pga,
                             "pga_gain_pct": 100 * (base - pga) / base,
                             "pga_wins": int(pga < base)})
    details = pd.DataFrame(rows)
    summary = details.groupby(["panel", "horizon", "metric", "baseline"], as_index=False).agg(
        markets_won=("pga_wins", "sum"), markets_compared=("pga_wins", "size"),
        mean_market_gain_pct=("pga_gain_pct", "mean"),
    )
    return details, summary


def wide_tables(market: pd.DataFrame, cross: pd.DataFrame, out: Path, order: list[str]):
    for (panel, h), group in market.groupby(["panel", "horizon"]):
        for metric in METRICS:
            table = group.pivot(index="node_name", columns="model", values=metric)
            node_order = group[["node", "node_name"]].drop_duplicates().sort_values("node").node_name
            table = table.reindex(index=node_order, columns=order)
            table.reset_index().rename(columns={"node_name": "Index"}).to_csv(
                out / f"table_{metric.lower()}_h{h}_{panel}.csv", index=False)
    cross.sort_values(["panel", "horizon", "model"]).to_csv(
        out / "cross_market_equal_weight_average.csv", index=False)


def main():
    args = options()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        raise FileExistsError(f"Comparison output already exists; use another --outdir: {args.outdir}")
    raw, canonical, models, variants, pga_cfg, base_cfg, ablation_meta = load_sources(args)
    ensemble = build_ensemble(raw, canonical)
    all_losses, alignment_audit = align_active_losses(ensemble, models)
    common_dates = set(
        all_losses.groupby(["horizon", "target_idx"]).node.nunique()
        .loc[lambda s: s == 8].index
    )
    common_keys = pd.MultiIndex.from_frame(all_losses[["horizon", "target_idx"]])
    common_losses = all_losses.loc[common_keys.isin(common_dates)].copy()
    if common_losses.empty:
        raise ValueError("No eight-market common target dates exist.")
    market_all, cross_all = summarize(all_losses, "all_days")
    market_common, cross_common = summarize(common_losses, "common_days")
    market = pd.concat([market_all, market_common], ignore_index=True)
    cross = pd.concat([cross_all, cross_common], ignore_index=True)
    seed_market, seed_market_summary, seed_cross_summary = seed_summaries(
        raw, all_losses, common_losses
    )
    order = list(MAIN_POOL) + ["Ablation-" + v for v in variants]
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw.groupby(["model", "horizon", "seed"], as_index=False).agg(
        rows=("active", "size"), active_rows=("active", "sum"),
        first_target=("target_date", "min"), last_target=("target_date", "max"),
    ).to_csv(out / "model_seed_inventory.csv", index=False)
    coverage = canonical.groupby("horizon", as_index=False).agg(
        full_model_active_market_dates=("active", "sum"),
        n_target_dates=("target_idx", "nunique"),
        first_target=("target_date", "min"), last_target=("target_date", "max"),
    )
    aligned_counts = all_losses.groupby("horizon", as_index=False).agg(
        aligned_market_dates=("node", lambda x: int(x.size // len(models)))
    )
    common_counts = common_losses.groupby("horizon", as_index=False).agg(
        eight_market_common_dates=("target_idx", "nunique")
    )
    coverage.merge(aligned_counts, on="horizon").merge(
        common_counts, on="horizon"
    ).to_csv(out / "aligned_coverage.csv", index=False)
    ensemble.to_csv(out / "seed_ensemble_predictions.csv", index=False)
    alignment_audit.to_csv(out / "alignment_audit.csv", index=False)
    market.to_csv(out / "market_metrics.csv", index=False)
    baseline_detail, baseline_summary = baseline_dominance(market)
    baseline_detail.to_csv(out / "dominance_vs_baselines.csv", index=False)
    baseline_summary.to_csv(out / "dominance_summary.csv", index=False)
    seed_market.to_csv(out / "per_seed_market_metrics.csv", index=False)
    seed_market_summary.to_csv(out / "per_seed_market_mean_std.csv", index=False)
    seed_cross_summary.to_csv(out / "per_seed_cross_market_mean_std.csv", index=False)
    wide_tables(market, cross, out, order)
    if variants:
        ablation_gains(market, variants).to_csv(out / "ablation_gains_vs_full.csv", index=False)
    if not args.skip_inference:
        pd.concat([
            dm_tables(all_losses, BASELINES).assign(panel="all_days"),
            dm_tables(common_losses, BASELINES).assign(panel="common_days"),
        ], ignore_index=True).to_csv(out / "dm_hac_vs_pga.csv", index=False)
        pool = models if args.mcs_include_ablations else list(MAIN_POOL)
        all_set, all_history = mcs_tables(all_losses, "all_days", pool, args)
        common_set, common_history = mcs_tables(common_losses, "common_days", pool, args)
        pd.concat([all_set, common_set], ignore_index=True).to_csv(out / "mcs_included_models.csv", index=False)
        pd.concat([all_history, common_history], ignore_index=True).to_csv(
            out / "mcs_elimination_history.csv", index=False)
    metadata = {
        "model_pool": models, "primary_mcs_pool": list(MAIN_POOL),
        "actual_mcs_pool": models if args.mcs_include_ablations else list(MAIN_POOL),
        "ablation_variants": variants, "horizons": args.horizons, "neural_seeds": args.seeds,
        "main_model_config": str((args.main_dir / "run_config.json").resolve()),
        "baseline_config": str((args.baseline_dir / "run_config.json").resolve()),
        "ablation_config": str((args.ablation_dir / "experiment_config.json").resolve()) if ablation_meta else None,
        "panel_sha256": sha256(args.panel_csv), "train_end": pga_cfg["train_end_exclusive"],
        "validation_end": pga_cfg["validation_end_exclusive"],
        "run_family": args.run_family,
        "primary_metric": "MSE on percent-volatility scale" if args.run_family == "mse" else "QLIKE on squared volatility",
        "reported_metrics": list(METRICS),
        "qlike_definition": "(y_vol/pred_vol)^2 - 2 log(y_vol/pred_vol) - 1",
        "seed_aggregation": "mean forecast first; score ensemble forecast second",
        "seed_stability": "individual seeds are also scored on the identical paired market-days; mean/std reported separately",
        "date_alignment": "exact target indices, target dates, origins, eight market identities, and observed targets; intersection of valid model-market-days",
        "common_days": "same already trained models restricted to targets with all eight markets active",
        "dm_direction": "competitor QLIKE/MSE/MAE minus full PGA; positive favors PGA",
        "dm_hac_bandwidth": "max(h-1, floor(4*(n/100)^(2/9)))",
        "dm_holm_family": "within horizon x market x metric x competitor pool",
        "mcs_alpha": args.mcs_alpha, "mcs_bootstrap": args.mcs_bootstrap,
        "mcs_block_length": "max(2,h,round(n^(1/3)))",
        "mcs_method": "range statistic with max_j studentized pairwise-loss elimination",
        "inference_skipped": args.skip_inference,
        "traditional_baselines": "OLS on volatility; QLIKE is evaluated out of sample",
        "pga_objective": pga_cfg.get("training_objective"),
        "baseline_neural_objective": base_cfg.get("neural_primary_loss"),
        "ablation_note": "Controlled variants are compared to a locked full model; legacy ladder variants change architecture and parameter count.",
    }
    (out / "comparison_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Compared {len(models)} models on {args.horizons} across eight markets.")
    print(f"Primary MCS pool: {len(MAIN_POOL)} models; ablation variants: {len(variants)}.")
    print(f"Saved QLIKE/MSE/MAE comparison to {out}")


if __name__ == "__main__":
    main()
