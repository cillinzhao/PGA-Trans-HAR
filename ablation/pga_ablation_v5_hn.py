#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
"""Nested PGA-Trans-HAR V5 ablation experiments for h=1, 5, 22.

Experiments
-----------
HAR + Temporal MHSA
HAR + Dynamic Prior Graph
HAR + MHSA + Dynamic Graph
Full PGA with fixed g

Every trainable ablation shares the same frozen HAR anchor, residual head,
dynamic residual gate r, robust MSE objective, rolling graph sequence and
60/10/30 validation/refit protocol. Components are added monotonically so
differences can be attributed to the named component.

Note: Pure HAR and Full PGA (dynamic_g) are excluded as they are evaluated in the main/baseline runs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 依赖于已经添加了 horizon 参数的 pga_trans_har_v5.py
import pga_trans_har_v5_hn_fixed as core


ABLATION_NAMES = {
    "temporal": "HAR+Temporal-MHSA",
    "dynamic_graph": "HAR+Dynamic-Prior-Graph",
    "temporal_dynamic_graph": "HAR+MHSA+Dynamic-Graph",
    "fixed_g": "PGA-Fixed-g",
}
TRAINABLE_ABLATIONS = tuple(ABLATION_NAMES.keys())


class TemporalOnlyBlock(nn.Module):
    def __init__(self, hidden_dim, num_nodes, seq_len, num_heads, dropout):
        super().__init__()
        self.temporal = core.MultiHeadTemporalAttention(
            hidden_dim, num_heads, dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, prior, return_gate=False):
        x = self.norm1(x + self.dropout(self.temporal(x, x_mask)))
        x = self.norm2(x + self.ffn(x))
        gate = torch.full(
            x.shape[:3], float("nan"), dtype=x.dtype, device=x.device
        )
        return x, (gate if return_gate else None)


class PriorGraphLayer(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, prior):
        B, S, N, H = x.shape
        active_source = x_mask[..., 0]
        graph = prior[:, None, :, :].expand(B, S, N, N)
        graph = graph * active_source[:, :, None, :]
        row_sum = graph.sum(dim=-1, keepdim=True)
        graph = graph / row_sum.clamp_min(1e-12)
        fallback = torch.eye(N, dtype=x.dtype, device=x.device)[None, None]
        graph = torch.where(row_sum > 1e-12, graph, fallback)
        value = self.value(x)
        aggregated = torch.einsum("bsij,bsjh->bsih", graph, value)
        return self.output(self.dropout(aggregated))


class PriorOnlyBlock(nn.Module):
    def __init__(self, hidden_dim, num_nodes, seq_len, num_heads, dropout):
        super().__init__()
        self.graph = PriorGraphLayer(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, x_mask, prior, return_gate=False):
        x = self.norm1(x + self.graph(x, x_mask, prior))
        x = self.norm2(x + self.ffn(x))
        gate = torch.zeros(x.shape[:3], dtype=x.dtype, device=x.device)
        return x, (gate if return_gate else None)


class TemporalPriorBlock(nn.Module):
    def __init__(self, hidden_dim, num_nodes, seq_len, num_heads, dropout):
        super().__init__()
        self.temporal = core.MultiHeadTemporalAttention(
            hidden_dim, num_heads, dropout
        )
        self.graph = PriorGraphLayer(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, prior, return_gate=False):
        x = self.norm1(x + self.dropout(self.temporal(x, x_mask)))
        x = self.norm2(x + self.graph(x, x_mask, prior))
        x = self.norm3(x + self.ffn(x))
        gate = torch.zeros(x.shape[:3], dtype=x.dtype, device=x.device)
        return x, (gate if return_gate else None)


class FixedGateSpatialAttention(core.DynamicPriorGuidedSpatialAttention):
    def __init__(self, hidden_dim, num_nodes, seq_len, dropout, fixed_g):
        super().__init__(hidden_dim, num_nodes, seq_len, dropout)
        if not 0.0 <= fixed_g <= 1.0:
            raise ValueError("fixed_g must be in [0,1].")
        self.fixed_g = float(fixed_g)

    def forward(self, x, x_mask, prior, return_gate=False):
        B, S, N, H = x.shape
        q = self.q(x).reshape(B * S, N, H)
        k = self.k(x).reshape(B * S, N, H)
        v = self.v(x).reshape(B * S, N, H)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        source_active = x_mask[..., 0].reshape(B * S, N)
        key_mask = source_active[:, None, :].expand(-1, N, -1)
        data_attention = core.masked_softmax(scores, key_mask)
        graph = prior[:, None].expand(B, S, N, N).reshape(B * S, N, N)
        graph = graph * key_mask.to(graph.dtype)
        row_sum = graph.sum(dim=-1, keepdim=True)
        graph = graph / row_sum.clamp_min(1e-12)
        graph = torch.where(row_sum > 1e-12, graph, data_attention)
        hybrid = self.fixed_g * data_attention + (1.0 - self.fixed_g) * graph
        output = torch.bmm(self.attention_dropout(hybrid), v)
        output = self.out(output.reshape(B, S, N, H))
        gate = torch.full(
            (B, S, N), self.fixed_g, dtype=x.dtype, device=x.device
        )
        return output, (gate if return_gate else None)


class FixedGateBlock(nn.Module):
    def __init__(
        self, hidden_dim, num_nodes, seq_len, num_heads, dropout, fixed_g
    ):
        super().__init__()
        self.temporal = core.MultiHeadTemporalAttention(
            hidden_dim, num_heads, dropout
        )
        self.spatial = FixedGateSpatialAttention(
            hidden_dim, num_nodes, seq_len, dropout, fixed_g
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mask, prior, return_gate=False):
        x = self.norm1(x + self.dropout(self.temporal(x, x_mask)))
        spatial, gate = self.spatial(x, x_mask, prior, return_gate)
        x = self.norm2(x + self.dropout(spatial))
        x = self.norm3(x + self.ffn(x))
        return x, gate


class AblationModel(core.PGATransHARV5):
    def __init__(self, ablation: str, fixed_g: float, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = kwargs.get("hidden_dim", 32)
        num_nodes = kwargs["num_nodes"]
        seq_len = kwargs.get("seq_len", 22)
        num_blocks = kwargs.get("num_blocks", 2)
        num_heads = kwargs.get("num_heads", 4)
        dropout = kwargs.get("dropout", 0.1)
        block_arguments = (hidden_dim, num_nodes, seq_len, num_heads, dropout)
        if ablation == "temporal":
            block_type = TemporalOnlyBlock
            self.blocks = nn.ModuleList(
                [block_type(*block_arguments) for _ in range(num_blocks)]
            )
        elif ablation == "dynamic_graph":
            block_type = PriorOnlyBlock
            self.blocks = nn.ModuleList(
                [block_type(*block_arguments) for _ in range(num_blocks)]
            )
        elif ablation == "temporal_dynamic_graph":
            block_type = TemporalPriorBlock
            self.blocks = nn.ModuleList(
                [block_type(*block_arguments) for _ in range(num_blocks)]
            )
        elif ablation == "fixed_g":
            self.blocks = nn.ModuleList(
                [
                    FixedGateBlock(*block_arguments, fixed_g=fixed_g)
                    for _ in range(num_blocks)
                ]
            )
        else:
            raise ValueError(f"Unknown trainable ablation: {ablation}")


def build_ablation_model(
    ablation,
    har_coefficients,
    config,
    fixed_g,
    num_nodes,
    seed,
    device,
):
    core.set_seed(seed)
    model = AblationModel(
        ablation=ablation,
        fixed_g=fixed_g,
        num_nodes=num_nodes,
        har_coefficients=har_coefficients,
        seq_len=config.seq_len,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_residual_scale=config.max_residual_scale,
    )
    return model.to(device)


def prediction_frame(model_name, seed, arrays, dates, names, horizon):
    frame = core.prediction_frame(arrays, dates, names, seed, horizon)
    frame["model"] = model_name
    return frame


def metric_frame(model_name, seed, arrays, names, horizon):
    rows = []
    for node, name in enumerate(names):
        active = arrays["mask"][:, node] > 0.5
        error = arrays["prediction"][active, node] - arrays["target"][active, node]
        rows.append(
            {
                "model": model_name,
                "horizon": int(horizon),
                "seed": int(seed),
                "node": node,
                "node_name": name,
                "n_active": int(active.sum()),
                "MSE": float(np.mean(error ** 2)),
                "MAE": float(np.mean(np.abs(error))),
            }
        )
    return pd.DataFrame(rows)


def ablation_gate_frame(model_name, seed, arrays, dates, names, alpha, horizon):
    attention = arrays["attention_gate"]
    last_gate = attention[-1, :, -1, :]
    finite = np.isfinite(attention[-1])
    gate_sum = np.where(finite, attention[-1], 0.0).sum(axis=1)
    gate_count = finite.sum(axis=1)
    window_gate = np.divide(
        gate_sum, gate_count,
        out=np.full_like(gate_sum, np.nan), where=gate_count > 0,
    )
    rows = []
    for row, target_idx in enumerate(arrays["target_idx"]):
        for node, name in enumerate(names):
            rows.append(
                {
                    "model": model_name,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "target_idx": int(target_idx),
                    "target_date": pd.Timestamp(dates[target_idx]).date().isoformat(),
                    "origin_idx": int(target_idx - horizon),
                    "origin_date": pd.Timestamp(dates[target_idx - horizon]).date().isoformat(),
                    "node": node,
                    "node_name": name,
                    "attention_gate_g": float(last_gate[row, node]),
                    "attention_gate_g_window_mean": float(window_gate[row, node]),
                    "residual_gate_r": float(arrays["residual_gate"][row, node]),
                    "node_residual_scale_alpha": float(alpha[node]),
                    "har_anchor": float(arrays["har_anchor"][row, node]),
                    "target_active": int(arrays["mask"][row, node] > 0.5),
                }
            )
    return pd.DataFrame(rows)


def summarize(metrics):
    return (
        metrics.groupby(["model", "horizon", "node", "node_name"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_active=("n_active", "min"),
            MSE_mean=("MSE", "mean"),
            MSE_std=("MSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
        )
        .fillna({"MSE_std": 0.0, "MAE_std": 0.0})
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="df_union_sqrt.csv", help="CSV path, e.g. df_union_sqrt.csv")
    parser.add_argument("--input-format", choices=["sqrt_rv", "rv"], default="sqrt_rv")
    parser.add_argument("--outdir", default="pga_ablation_results_v5")
    parser.add_argument(
        "--ablations", nargs="+", choices=list(ABLATION_NAMES), default=list(ABLATION_NAMES)
    )
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22])  # [新增] 支持多步预测
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seq-len", type=int, default=22)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-residual-scale", type=float, default=1.0)
    parser.add_argument("--robust-lambda", type=float, default=0.10)
    parser.add_argument("--robust-temperature", type=float, default=0.20)
    parser.add_argument("--fixed-g", type=float, default=0.50)
    parser.add_argument("--graph-window", type=int, default=252)
    parser.add_argument("--graph-update-every", type=int, default=20)
    parser.add_argument("--var-lag", type=int, default=2)
    parser.add_argument("--var-ridge", type=float, default=5.0)
    parser.add_argument("--dy-horizon", type=int, default=22)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args, unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    args.horizons = sorted(set(args.horizons))
    if not args.horizons or any(horizon < 1 for horizon in args.horizons):
        raise ValueError("--horizons must contain positive integers.")
    if args.seq_len != 22:
        raise ValueError("The V5 ablation protocol fixes seq_len=22.")
    panel = core.load_panel(args.data, args.input_format)
    values, dates, names = panel.values, panel.dates, panel.names
    train_end, val_end = core.fixed_split(
        len(values), args.train_ratio, args.val_ratio
    )
    device = core.choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    config = core.TrainConfig(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_residual_scale=args.max_residual_scale,
        robust_lambda=args.robust_lambda,
        robust_temperature=args.robust_temperature,
    )

    global_frames, global_metrics, global_histories, global_gates, global_inventory = [], [], [], [], []

    # --- 外层遍历 Horizon ---
    for h in args.horizons:
        print(f"\n{'='*60}")
        print(f"=== Starting Ablation Horizon: h = {h} ===")
        print(f"{'='*60}")

        validation_target_start = train_end + h - 1
        test_target_start = val_end + h - 1

        # 构建支持 horizon 的 Windows
        train_windows = core.make_windows(values, args.seq_len, train_end, args.seq_len, horizon=h)
        val_windows = core.make_windows(
            values, validation_target_start, val_end, args.seq_len, horizon=h
        )
        refit_windows = core.make_windows(values, args.seq_len, val_end, args.seq_len, horizon=h)
        test_windows = core.make_windows(
            values, test_target_start, len(values), args.seq_len, horizon=h
        )
        if int(val_windows[4][0]) - h != train_end - 1:
            raise AssertionError("Validation forecast origin is not split-safe.")
        if int(test_windows[4][0]) - h != val_end - 1:
            raise AssertionError("Test forecast origin is not split-safe.")
        
        all_target_indices = np.unique(np.concatenate([
            train_windows[4], val_windows[4], refit_windows[4], test_windows[4]
        ]))
        
        all_priors, graph_diagnostics = core.build_dynamic_priors(
            values,
            all_target_indices,
            args.seq_len,
            args.graph_window,
            args.graph_update_every,
            args.var_lag,
            args.var_ridge,
            args.dy_horizon,
            forecast_horizon=h
        )
        prior_lookup = {int(i): p for i, p in zip(all_target_indices, all_priors)}

        def dataset(window_tuple):
            priors = np.asarray([prior_lookup[int(i)] for i in window_tuple[-1]])
            return core.ForecastDataset(window_tuple, priors)

        train_data, val_data = dataset(train_windows), dataset(val_windows)
        refit_data, test_data = dataset(refit_windows), dataset(test_windows)
        val_loader = core.make_loader(val_data, config.batch_size, False)
        test_loader = core.make_loader(test_data, config.batch_size, False)
        
        # 使用特定 horizon 计算 HAR anchor 和 MSE reference
        har_dev = core.fit_har_anchor(values, train_end, args.seq_len, horizon=h)
        scale_dev, har_mse_dev = core.training_statistics(
            values, train_end, args.seq_len, har_dev, horizon=h
        )
        har_refit = core.fit_har_anchor(values, val_end, args.seq_len, horizon=h)
        scale_refit, har_mse_refit = core.training_statistics(
            values, val_end, args.seq_len, har_refit, horizon=h
        )
        
        h_dir = outdir / f"h{h}"
        h_dir.mkdir(parents=True, exist_ok=True)
        graph_diagnostics.to_csv(h_dir / f"dynamic_graph_diagnostics_h{h}.csv", index=False)

        # 进行所有配置的训练验证
        for ablation in TRAINABLE_ABLATIONS:
            if ablation not in args.ablations:
                continue
            model_name = ABLATION_NAMES[ablation]
            for seed in args.seeds:
                print(f"\n--- [h={h}] {model_name}, seed {seed} ---", flush=True)
                dev_objective = core.RobustRelativeMSE(
                    scale_dev,
                    har_mse_dev,
                    config.robust_lambda,
                    config.robust_temperature,
                ).to(device)
                
                model = build_ablation_model(
                    ablation, har_dev, config, args.fixed_g, len(names), seed, device
                )
                train_loader = core.make_loader(train_data, config.batch_size, True)
                model, selected_epoch, history = core.select_epoch_on_validation(
                    model,
                    train_loader,
                    val_loader,
                    dev_objective,
                    scale_dev,
                    har_mse_dev,
                    config,
                    device,
                )
                history.insert(0, "model", model_name)
                history.insert(1, "ablation", ablation)
                history.insert(2, "seed", seed)
                history.insert(3, "horizon", h)
                history["selected_epoch"] = selected_epoch
                global_histories.append(history)

                refit_objective = core.RobustRelativeMSE(
                    scale_refit,
                    har_mse_refit,
                    config.robust_lambda,
                    config.robust_temperature,
                ).to(device)
                
                final_model = build_ablation_model(
                    ablation, har_refit, config, args.fixed_g, len(names), seed, device
                )
                final_model, final_train_loss = core.refit_fixed_epochs(
                    final_model,
                    core.make_loader(refit_data, config.batch_size, True),
                    refit_objective,
                    selected_epoch,
                    config,
                    device,
                )
                arrays = core.collect_predictions(
                    final_model, test_loader, device, return_aux=True
                )
                frame = prediction_frame(model_name, seed, arrays, dates, names, horizon=h)
                metrics = metric_frame(model_name, seed, arrays, names, horizon=h)
                metrics["selected_epoch"] = selected_epoch
                metrics["final_train_objective"] = final_train_loss
                alpha = (
                    config.max_residual_scale
                    * torch.tanh(final_model.alpha_raw.detach())
                ).cpu().numpy()[:, 0]
                gate = ablation_gate_frame(
                    model_name, seed, arrays, dates, names, alpha, horizon=h
                )
                
                model_dir = h_dir / ablation / f"seed{seed}"
                model_dir.mkdir(parents=True, exist_ok=True)
                frame.to_csv(
                    model_dir / f"{ablation}_h{h}_seed{seed}_predictions.csv", index=False
                )
                metrics.to_csv(model_dir / f"metrics_h{h}_seed{seed}.csv", index=False)
                gate.to_csv(model_dir / f"gates_h{h}_seed{seed}.csv", index=False)
                torch.save(
                    {
                        "model_state_dict": final_model.state_dict(),
                        "model": model_name,
                        "ablation": ablation,
                        "seed": seed,
                        "selected_epoch": selected_epoch,
                        "horizon": h,
                    },
                    model_dir / f"model_h{h}_seed{seed}.pt",
                )
                
                global_frames.append(frame)
                global_metrics.append(metrics)
                global_gates.append(gate)
                global_inventory.append(
                    {
                        "model": model_name,
                        "ablation": ablation,
                        "horizon": h,
                        "seed": seed,
                        "parameter_count": sum(
                            p.numel() for p in final_model.parameters() if p.requires_grad
                        ),
                        "selected_epoch": selected_epoch,
                    }
                )

    if not global_frames:
        raise RuntimeError("No ablation models were run.")

    metrics = pd.concat(global_metrics, ignore_index=True)
    predictions = pd.concat(global_frames, ignore_index=True)
    summary = summarize(metrics)
    
    metrics.to_csv(outdir / "all_seed_metrics_ALL_HORIZONS.csv", index=False)
    predictions.to_csv(outdir / "all_seed_predictions_ALL_HORIZONS.csv", index=False)
    summary.to_csv(outdir / "summary_mean_std_ALL_HORIZONS.csv", index=False)
    pd.DataFrame(global_inventory).to_csv(outdir / "model_inventory.csv", index=False)
    
    if global_histories:
        pd.concat(global_histories, ignore_index=True).to_csv(
            outdir / "validation_history_ALL_HORIZONS.csv", index=False
        )
    if global_gates:
        pd.concat(global_gates, ignore_index=True).to_csv(
            outdir / "all_seed_gates_ALL_HORIZONS.csv", index=False
        )
        
    with open(outdir / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                **vars(args),
                "ablation_model_names": ABLATION_NAMES,
                "horizons": args.horizons,
                "rows": len(values),
                "nodes": len(names),
                "zero_codes_restored_to_nan": panel.zero_count,
                "train_end_exclusive": train_end,
                "validation_end_exclusive": val_end,
                "test_first_target_date_by_horizon": {
                    str(h): dates[val_end + h - 1].date().isoformat()
                    for h in args.horizons
                },
                "test_split_first_date": dates[val_end].date().isoformat(),
                "test_last_date": dates[-1].date().isoformat(),
                "test_used_for_selection": False,
                "refit_on_first_70_percent": True,
                "train_config": asdict(config),
                "reported_metrics": ["MSE", "MAE"],
                "component_control": (
                    "All trainable ablations share HAR anchor, residual head, "
                    "residual gate r and robust MSE; only named graph/MHSA/g components vary."
                ),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
        
    print("\n" + "="*60)
    print("ALL ABLATIONS AND HORIZONS DONE")
    print("="*60)
    print("\nV5 ablation mean/std")
    print(summary.to_string(index=False))
    print(f"\nSaved V5 ablations to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


# In[ ]:




