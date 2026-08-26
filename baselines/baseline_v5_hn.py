#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import json
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import GraphicalLasso
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# 依赖于已经添加了 horizon 参数的 pga_trans_har_v5.py
import pga_trans_har_v5_hn_fixed as protocol


TRADITIONAL_MODELS = ("HAR", "VHAR", "HAR-KS")
DEEP_MODELS = ("GNN-HAR", "DCRNN-HAR")
ALL_MODELS = (*TRADITIONAL_MODELS, *DEEP_MODELS)


# ---------------------------------------------------------------------------
# Shared mask-aware HAR features
# ---------------------------------------------------------------------------


def panel_features(window_tuple):
    X, Xm, Y, Ym, indices = window_tuple
    overlap, valid = protocol.har_features_numpy(X, Xm)
    values = X[..., 0]
    active = Xm[..., 0] > 0.5
    S = values.shape[1]
    daily, daily_ok = protocol.numpy_last_valid(values, active)
    previous_week, week_ok = protocol.numpy_masked_mean(
        values, active, max(0, S - 5), S - 1
    )
    previous_month, month_ok = protocol.numpy_masked_mean(
        values, active, 0, max(1, S - 5)
    )
    nonoverlap = np.stack([daily, previous_week, previous_month], axis=-1)
    nonoverlap_valid = daily_ok & week_ok & month_ok
    vhar = overlap.reshape(len(overlap), -1)
    vhar_valid = valid.all(axis=1)
    N = overlap.shape[1]
    kitchen = np.zeros((len(overlap), N, N + 2), dtype=np.float32)
    kitchen_valid = np.zeros((len(overlap), N), dtype=bool)
    for node in range(N):
        other_daily = np.delete(daily, node, axis=1)
        other_ok = np.delete(daily_ok, node, axis=1).all(axis=1)
        kitchen[:, node] = np.concatenate(
            [overlap[:, node], other_daily], axis=1
        )
        kitchen_valid[:, node] = valid[:, node] & other_ok
    return {
        "overlap": overlap,
        "overlap_valid": valid,
        "nonoverlap": nonoverlap.astype(np.float32),
        "nonoverlap_valid": nonoverlap_valid,
        "vhar": vhar.astype(np.float32),
        "vhar_valid": vhar_valid,
        "kitchen": kitchen,
        "kitchen_valid": kitchen_valid,
        "target": Y[..., 0],
        "target_mask": Ym[..., 0] > 0.5,
        "target_idx": indices,
    }


def fit_ols(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    augmented = np.column_stack([np.ones(len(design)), design])
    coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
    return coefficients


def predict_ols(design: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(design)), design]) @ coefficients


def run_traditional(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    names: Sequence[str],
    val_end: int,
    seq_len: int,
    outdir: Path,
    selected_models: Sequence[str],
    horizon: int,  # [新增]
):
    # 传入 horizon 防止未来数据泄露
    all_windows = protocol.make_windows(values, seq_len, len(values), seq_len, horizon=horizon)
    features = panel_features(all_windows)
    train = features["target_idx"] < val_end
    # At horizon h, the earliest leakage-free test target is val_end+h-1;
    # its forecast origin is exactly val_end-1, the final refit observation.
    test_target_start = val_end + horizon - 1
    test = features["target_idx"] >= test_target_start
    target = features["target"]
    target_mask = features["target_mask"]
    frames, metric_frames = [], []
    N = len(names)

    for model_name in TRADITIONAL_MODELS:
        if model_name not in selected_models:
            continue
        prediction = np.full((int(test.sum()), N), np.nan, dtype=np.float64)
        evaluation_mask = np.zeros_like(prediction, dtype=bool)
        for node in range(N):
            if model_name == "HAR":
                design = features["overlap"][:, node]
                feature_ok = features["overlap_valid"][:, node]
            elif model_name == "VHAR":
                design = features["vhar"]
                feature_ok = features["vhar_valid"]
            else:
                design = features["kitchen"][:, node]
                feature_ok = features["kitchen_valid"][:, node]
            fit_rows = train & feature_ok & target_mask[:, node]
            if fit_rows.sum() <= design.shape[1] + 10:
                raise RuntimeError(f"Insufficient {model_name} rows for node {node}.")
            coefficients = fit_ols(design[fit_rows], target[fit_rows, node])
            test_ok = feature_ok[test]
            prediction[test_ok, node] = predict_ols(
                design[test][test_ok], coefficients
            )
            evaluation_mask[:, node] = target_mask[test, node] & test_ok

        prediction = np.maximum(prediction, protocol.EPS)
        arrays = {
            "prediction": prediction,
            "target": target[test],
            "mask": evaluation_mask.astype(np.float32),
            "target_idx": features["target_idx"][test],
        }
        frame = canonical_prediction_frame(
            model_name, 0, arrays, dates, names, horizon=horizon
        )
        metrics = metric_table(model_name, 0, arrays, names, horizon=horizon)
        model_dir = outdir / f"h{horizon}" / model_name / "seed0"
        model_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(
            model_dir / f"{model_name}_h{horizon}_seed0_predictions.csv", index=False
        )
        metrics.to_csv(model_dir / f"metrics_h{horizon}_seed0.csv", index=False)
        frames.append(frame)
        metric_frames.append(metrics)
    return frames, metric_frames


# ---------------------------------------------------------------------------
# Static graphs for the deep baselines
# ---------------------------------------------------------------------------


def complete_training_panel(values: np.ndarray) -> np.ndarray:
    return protocol.causal_fill_window(values)


def glasso_adjacency(values: np.ndarray, alpha: float) -> np.ndarray:
    complete = complete_training_panel(values)
    z = (complete - complete.mean(axis=0)) / np.maximum(
        complete.std(axis=0, ddof=1), 1e-8
    )
    precision = GraphicalLasso(alpha=alpha, max_iter=2000).fit(z).precision_
    denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    adjacency = np.divide(
        np.abs(-precision),
        denominator,
        out=np.zeros_like(precision),
        where=denominator > 0,
    )
    np.fill_diagonal(adjacency, 1.0)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1e-12)
    return adjacency.astype(np.float32)


def static_dy_adjacency(
    values: np.ndarray, var_lag: int, ridge: float, horizon: int
) -> np.ndarray:
    return protocol.ridge_var_gfevd(
        values, lags=var_lag, ridge=ridge, horizon=horizon
    )


# ---------------------------------------------------------------------------
# Deep baseline datasets and models
# ---------------------------------------------------------------------------


class GNNDataset(Dataset):
    def __init__(self, nonoverlap, overlap, target, target_mask, indices):
        self.nonoverlap = torch.as_tensor(nonoverlap, dtype=torch.float32)
        self.overlap = torch.as_tensor(overlap, dtype=torch.float32)
        self.target = torch.as_tensor(target[..., None], dtype=torch.float32)
        self.target_mask = torch.as_tensor(
            target_mask[..., None], dtype=torch.float32
        )
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        return (
            self.nonoverlap[index],
            self.overlap[index],
            self.target[index],
            self.target_mask[index],
            self.indices[index],
        )


class SequenceDataset(Dataset):
    def __init__(self, window_tuple):
        X, Xm, Y, Ym, indices = window_tuple
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Xm = torch.as_tensor(Xm, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.Ym = torch.as_tensor(Ym, dtype=torch.float32)
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Xm[index], self.Y[index], self.Ym[index], self.indices[index]


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency):
        return torch.relu(self.linear(torch.einsum("ij,bjf->bif", adjacency, x)))


class GNNHAR(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GraphConv(3, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, 1)
        self.har = nn.ModuleList([nn.Linear(3, 1) for _ in range(num_nodes)])

    def forward(self, nonoverlap, overlap, adjacency):
        graph = self.gcn2(self.gcn1(nonoverlap, adjacency), adjacency)
        har = torch.cat(
            [self.har[node](overlap[:, node]) for node in range(self.num_nodes)],
            dim=1,
        )[..., None]
        return F.softplus(har + graph).clamp_min(protocol.EPS)


def masked_row_normalize(adjacency: torch.Tensor, source_active: torch.Tensor):
    masked = adjacency[None, :, :] * source_active[:, None, :]
    row_sum = masked.sum(dim=-1, keepdim=True)
    fallback = torch.eye(
        adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype
    )[None].expand(masked.size(0), -1, -1)
    normalized = masked / row_sum.clamp_min(1e-12)
    return torch.where(row_sum > 1e-12, normalized, fallback)


class DiffusionConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, steps: int):
        super().__init__()
        self.steps = steps
        self.linear = nn.Linear(input_dim * (2 * steps + 1), output_dim)

    def forward(self, x, source_active, adjacency):
        forward = masked_row_normalize(adjacency, source_active)
        reverse = masked_row_normalize(adjacency.t(), source_active)
        outputs, xf, xr = [x], x, x
        for _ in range(self.steps):
            xf = torch.bmm(forward, xf)
            xr = torch.bmm(reverse, xr)
            outputs.extend([xf, xr])
        return self.linear(torch.cat(outputs, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, diffusion_steps: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = DiffusionConv(
            input_dim + hidden_dim, hidden_dim * 2, diffusion_steps
        )
        self.candidate = DiffusionConv(
            input_dim + hidden_dim, hidden_dim, diffusion_steps
        )

    def forward(self, x, hidden, source_active, adjacency):
        joined = torch.cat([x, hidden], dim=-1)
        reset, update = torch.chunk(
            torch.sigmoid(self.gates(joined, source_active, adjacency)), 2, dim=-1
        )
        candidate = torch.tanh(
            self.candidate(
                torch.cat([x, reset * hidden], dim=-1),
                source_active,
                adjacency,
            )
        )
        return update * hidden + (1.0 - update) * candidate


class DCRNNHAR(nn.Module):
    def __init__(
        self, num_nodes: int, seq_len: int, hidden_dim: int = 16, diffusion_steps: int = 2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.cell = DCGRUCell(1, hidden_dim, diffusion_steps)
        self.projection = nn.Linear(hidden_dim, 1)
        self.har = nn.ModuleList([nn.Linear(3, 1) for _ in range(num_nodes)])

    def forward(self, x, x_mask, adjacency):
        hidden = torch.zeros(
            x.size(0), self.num_nodes, self.hidden_dim, device=x.device
        )
        for time in range(self.seq_len):
            hidden = self.cell(
                x[:, time], hidden, x_mask[:, time, :, 0], adjacency
            )
        recurrent = self.projection(hidden)
        daily = protocol.torch_last_valid(x, x_mask)
        weekly = protocol.torch_masked_mean(x, x_mask, self.seq_len - 5, self.seq_len)
        monthly = protocol.torch_masked_mean(x, x_mask, 0, self.seq_len)
        features = torch.cat([daily, weekly, monthly], dim=-1)
        har = torch.cat(
            [self.har[node](features[:, node]) for node in range(self.num_nodes)],
            dim=1,
        )[..., None]
        return F.softplus(recurrent + har).clamp_min(protocol.EPS)


class NodeBalancedMSE(nn.Module):
    def __init__(self, node_scale: np.ndarray):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(node_scale).view(1, -1))

    def forward(self, prediction, target, mask):
        error2 = (prediction[..., 0] - target[..., 0]) ** 2
        active = mask[..., 0]
        count = active.sum(dim=0)
        valid = count > 0
        node_mse = (error2 * active).sum(dim=0) / count.clamp_min(1.0)
        standardized = node_mse / self.scale[0].square().clamp_min(1e-8)
        return standardized[valid].mean()


def target_scales(values: np.ndarray, target_end: int, seq_len: int, horizon: int):
    # [新增 horizon 参数传入]
    windows = protocol.make_windows(values, seq_len, target_end, seq_len, horizon=horizon)
    y, mask = windows[2][..., 0], windows[3][..., 0] > 0.5
    scales = []
    for node in range(y.shape[1]):
        scales.append(max(float(np.std(y[mask[:, node], node], ddof=1)), 1e-3))
    return np.asarray(scales, dtype=np.float32)


def loader(dataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )


def run_forward(model, batch, adjacency, model_name, device):
    if model_name == "GNN-HAR":
        nonoverlap, overlap, target, mask, indices = batch
        prediction = model(
            nonoverlap.to(device), overlap.to(device), adjacency
        )
    else:
        x, x_mask, target, mask, indices = batch
        prediction = model(x.to(device), x_mask.to(device), adjacency)
    return prediction, target.to(device), mask.to(device), indices


def train_epoch(model, data_loader, objective, optimizer, adjacency, model_name, device):
    model.train()
    total, batches = 0.0, 0
    for batch in data_loader:
        optimizer.zero_grad(set_to_none=True)
        prediction, target, mask, _ = run_forward(
            model, batch, adjacency, model_name, device
        )
        loss = objective(prediction, target, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def evaluate_objective(model, data_loader, objective, adjacency, model_name, device):
    model.eval()
    predictions, targets, masks = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            prediction, target, mask, _ = run_forward(
                model, batch, adjacency, model_name, device
            )
            predictions.append(prediction)
            targets.append(target)
            masks.append(mask)
    return float(
        objective(
            torch.cat(predictions), torch.cat(targets), torch.cat(masks)
        ).item()
    )


def select_epoch(
    model,
    train_loader,
    val_loader,
    objective,
    adjacency,
    model_name,
    device,
    max_epochs,
    patience,
    learning_rate,
):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )
    best_value, best_epoch, wait, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, max_epochs + 1):
        train_value = train_epoch(
            model, train_loader, objective, optimizer, adjacency, model_name, device
        )
        val_value = evaluate_objective(
            model, val_loader, objective, adjacency, model_name, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_node_balanced_mse": train_value,
                "val_node_balanced_mse": val_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if val_value < best_value - 1e-8:
            best_value, best_epoch, wait = val_value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        scheduler.step()
        if wait >= patience:
            break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")
    model.load_state_dict(best_state)
    return model, best_epoch, pd.DataFrame(history)


def refit(model, data_loader, objective, adjacency, model_name, device, epochs, lr):
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-6
    )
    final_value = float("nan")
    for _ in range(epochs):
        final_value = train_epoch(
            model, data_loader, objective, optimizer, adjacency, model_name, device
        )
        scheduler.step()
    return model, final_value


def predict(model, data_loader, adjacency, model_name, device):
    model.eval()
    predictions, targets, masks, indices = [], [], [], []
    with torch.no_grad():
        for batch in data_loader:
            prediction, target, mask, index = run_forward(
                model, batch, adjacency, model_name, device
            )
            predictions.append(prediction.cpu().numpy()[..., 0])
            targets.append(target.cpu().numpy()[..., 0])
            masks.append(mask.cpu().numpy()[..., 0])
            indices.append(index.numpy())
    return {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "mask": np.concatenate(masks),
        "target_idx": np.concatenate(indices),
    }


def build_gnn_dataset(window_tuple):
    features = panel_features(window_tuple)
    return GNNDataset(
        features["nonoverlap"],
        features["overlap"],
        features["target"],
        features["target_mask"],
        features["target_idx"],
    )


def construct_model(model_name, num_nodes, seq_len, seed, device):
    protocol.set_seed(seed)
    if model_name == "GNN-HAR":
        model = GNNHAR(num_nodes)
    else:
        model = DCRNNHAR(num_nodes, seq_len)
    return model.to(device)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def canonical_prediction_frame(model, seed, arrays, dates, names, horizon):
    rows = []
    for row, target_idx in enumerate(arrays["target_idx"]):
        for node, name in enumerate(names):
            active = bool(arrays["mask"][row, node] > 0.5)
            rows.append(
                {
                    "model": model,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "target_idx": int(target_idx),
                    "target_date": pd.Timestamp(dates[target_idx]).date().isoformat(),
                    "origin_idx": int(target_idx - horizon),
                    "origin_date": pd.Timestamp(dates[target_idx - horizon]).date().isoformat(),
                    "node": node,
                    "node_name": name,
                    "y_sqrt_rv_pct": float(arrays["target"][row, node]) if active else np.nan,
                    "pred_sqrt_rv_pct": float(arrays["prediction"][row, node]),
                    "active": int(active),
                }
            )
    return pd.DataFrame(rows)


def metric_table(model, seed, arrays, names, horizon):
    rows = []
    for node, name in enumerate(names):
        active = arrays["mask"][:, node] > 0.5
        error = arrays["prediction"][active, node] - arrays["target"][active, node]
        rows.append(
            {
                "model": model,
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


def run_deep_model_seed(
    model_name,
    seed,
    values,
    dates,
    names,
    train_end,
    val_end,
    config,
    graph_config,
    device,
    outdir,
    horizon,  # [新增]
):
    # 全部透传 horizon 参数
    validation_target_start = train_end + horizon - 1
    test_target_start = val_end + horizon - 1
    train_windows = protocol.make_windows(values, config.seq_len, train_end, config.seq_len, horizon=horizon)
    val_windows = protocol.make_windows(
        values, validation_target_start, val_end, config.seq_len, horizon=horizon
    )
    refit_windows = protocol.make_windows(values, config.seq_len, val_end, config.seq_len, horizon=horizon)
    test_windows = protocol.make_windows(
        values, test_target_start, len(values), config.seq_len, horizon=horizon
    )
    if int(val_windows[4][0]) - horizon != train_end - 1:
        raise AssertionError("Validation forecast origin is not split-safe.")
    if int(test_windows[4][0]) - horizon != val_end - 1:
        raise AssertionError("Test forecast origin is not split-safe.")
    
    dataset_factory = build_gnn_dataset if model_name == "GNN-HAR" else SequenceDataset
    train_data = dataset_factory(train_windows)
    val_data = dataset_factory(val_windows)
    refit_data = dataset_factory(refit_windows)
    test_data = dataset_factory(test_windows)
    
    scale_dev = target_scales(values, train_end, config.seq_len, horizon=horizon)
    scale_refit = target_scales(values, val_end, config.seq_len, horizon=horizon)
    objective_dev = NodeBalancedMSE(scale_dev).to(device)
    objective_refit = NodeBalancedMSE(scale_refit).to(device)

    # 静态图只在训练集上构建，不受 horizon 影响（它反映的是整体相关性），原样保留即可
    graph_function = glasso_adjacency if model_name == "GNN-HAR" else static_dy_adjacency
    if model_name == "GNN-HAR":
        graph_dev_np = graph_function(values[:train_end], graph_config.glasso_alpha)
        graph_refit_np = graph_function(values[:val_end], graph_config.glasso_alpha)
    else:
        graph_dev_np = graph_function(
            values[:train_end], graph_config.var_lag, graph_config.var_ridge, graph_config.dy_horizon
        )
        graph_refit_np = graph_function(
            values[:val_end], graph_config.var_lag, graph_config.var_ridge, graph_config.dy_horizon
        )
    graph_dev = torch.as_tensor(graph_dev_np, dtype=torch.float32, device=device)
    graph_refit = torch.as_tensor(graph_refit_np, dtype=torch.float32, device=device)

    lr = config.gnn_lr if model_name == "GNN-HAR" else config.dcrnn_lr
    model = construct_model(model_name, len(names), config.seq_len, seed, device)
    model, selected_epoch, history = select_epoch(
        model,
        loader(train_data, config.batch_size, True, seed),
        loader(val_data, config.batch_size, False, seed),
        objective_dev,
        graph_dev,
        model_name,
        device,
        config.max_epochs,
        config.patience,
        lr,
    )
    history.insert(0, "model", model_name)
    history.insert(1, "seed", seed)
    history["selected_epoch"] = selected_epoch

    final_model = construct_model(model_name, len(names), config.seq_len, seed, device)
    final_model, final_train_value = refit(
        final_model,
        loader(refit_data, config.batch_size, True, seed),
        objective_refit,
        graph_refit,
        model_name,
        device,
        selected_epoch,
        lr,
    )
    arrays = predict(
        final_model,
        loader(test_data, config.batch_size, False, seed),
        graph_refit,
        model_name,
        device,
    )
    frame = canonical_prediction_frame(model_name, seed, arrays, dates, names, horizon=horizon)
    metrics = metric_table(model_name, seed, arrays, names, horizon=horizon)
    metrics["selected_epoch"] = selected_epoch
    metrics["final_train_objective"] = final_train_value
    
    model_dir = outdir / f"h{horizon}" / model_name / f"seed{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        model_dir / f"{model_name}_h{horizon}_seed{seed}_predictions.csv", index=False
    )
    metrics.to_csv(model_dir / f"metrics_h{horizon}_seed{seed}.csv", index=False)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "selected_epoch": selected_epoch,
            "seed": seed,
            "model": model_name,
        },
        model_dir / f"model_h{horizon}_seed{seed}.pt",
    )
    inventory = {
        "model": model_name,
        "seed": seed,
        "selected_epoch": selected_epoch,
        "parameter_count": sum(
            p.numel() for p in final_model.parameters() if p.requires_grad
        ),
    }
    return frame, metrics, history, inventory


@dataclass
class DeepConfig:
    seq_len: int = 22
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 15
    gnn_lr: float = 0.002
    dcrnn_lr: float = 0.001


@dataclass
class GraphConfig:
    glasso_alpha: float = 0.08
    var_lag: int = 2
    var_ridge: float = 5.0
    dy_horizon: int = 22


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="df_union_sqrt.csv")
    parser.add_argument("--input-format", choices=["sqrt_rv", "rv"], default="sqrt_rv")
    parser.add_argument("--outdir", default="baseline_results_v5")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22]) # [新增 horizons 列表]
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seq-len", type=int, default=22)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--gnn-lr", type=float, default=0.002)
    parser.add_argument("--dcrnn-lr", type=float, default=0.001)
    parser.add_argument("--glasso-alpha", type=float, default=0.08)
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
        raise ValueError("Baseline V5 fixes seq_len=22.")
        
    panel = protocol.load_panel(args.data, args.input_format)
    values, dates, names = panel.values, panel.dates, panel.names
    train_end, val_end = protocol.fixed_split(
        len(values), args.train_ratio, args.val_ratio
    )
    device = protocol.choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    deep_config = DeepConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        gnn_lr=args.gnn_lr,
        dcrnn_lr=args.dcrnn_lr,
    )
    graph_config = GraphConfig(
        glasso_alpha=args.glasso_alpha,
        var_lag=args.var_lag,
        var_ridge=args.var_ridge,
        dy_horizon=args.dy_horizon,
    )

    frames, metric_frames, histories = [], [], []
    inventory_dict = {}

    # 外层套入 Horizon 循环
    for h in args.horizons:
        print(f"\n{'='*60}")
        print(f"=== Starting Baseline Horizon: h = {h} ===")
        print(f"{'='*60}")

        # 1. 跑传统的计量基线模型 (HAR, VHAR, HAR-KS)
        traditional_frames, traditional_metrics = run_traditional(
            values, dates, names, val_end, args.seq_len, outdir, args.models, horizon=h
        )
        frames.extend(traditional_frames)
        metric_frames.extend(traditional_metrics)
        
        # 计入库存 (库存信息对模型本身是通用的，跟 horizon 无关)
        for model_name in TRADITIONAL_MODELS:
            if model_name in args.models:
                inventory_dict[model_name] = {
                    "model": model_name,
                    "seed": 0,
                    "selected_epoch": 0,
                    "parameter_count": 0,
                }

        # 2. 跑深度基线模型 (GNN-HAR, DCRNN-HAR)
        for model_name in DEEP_MODELS:
            if model_name not in args.models:
                continue
            for seed in args.seeds:
                print(f"\n--- [h={h}] {model_name}, seed {seed} ---", flush=True)
                frame, metrics, history, item = run_deep_model_seed(
                    model_name,
                    seed,
                    values,
                    dates,
                    names,
                    train_end,
                    val_end,
                    deep_config,
                    graph_config,
                    device,
                    outdir,
                    horizon=h,
                )
                frames.append(frame)
                metric_frames.append(metrics)
                histories.append(history)
                inventory_dict[f"{model_name}_seed{seed}"] = item

    if not frames:
        raise RuntimeError("No baseline models were selected.")
        
    metrics = pd.concat(metric_frames, ignore_index=True)
    predictions = pd.concat(frames, ignore_index=True)
    summary = summarize(metrics)
    inventory = list(inventory_dict.values())
    
    metrics.to_csv(outdir / "all_seed_metrics_ALL_HORIZONS.csv", index=False)
    predictions.to_csv(outdir / "all_seed_predictions_ALL_HORIZONS.csv", index=False)
    summary.to_csv(outdir / "summary_mean_std_ALL_HORIZONS.csv", index=False)
    pd.DataFrame(inventory).to_csv(outdir / "model_inventory.csv", index=False)
    
    if histories:
        pd.concat(histories, ignore_index=True).to_csv(
            outdir / "validation_history_ALL_HORIZONS.csv", index=False
        )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                **vars(args),
                "rows": len(values),
                "nodes": len(names),
                "zero_codes_restored_to_nan": panel.zero_count,
                "horizons": args.horizons,
                "train_end_exclusive": train_end,
                "validation_end_exclusive": val_end,
                "test_first_target_date_by_horizon": {
                    str(h): dates[val_end + h - 1].date().isoformat()
                    for h in args.horizons
                },
                "test_split_first_date": dates[val_end].date().isoformat(),
                "test_last_date": dates[-1].date().isoformat(),
                "split_protocol": "60_train_10_validation_30_test",
                "deep_refit_on_first_70_percent": True,
                "test_used_for_selection": False,
                "deep_training_objective": "node_balanced_MSE",
                "reported_metrics": ["MSE", "MAE"],
                "deep_config": asdict(deep_config),
                "graph_config": asdict(graph_config),
                "device_used": str(device),
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
        
    print("\n" + "="*60)
    print("ALL HORIZONS DONE (BASELINES)")
    print("="*60)
    print("\nBaseline V5 mean/std")
    print(summary.to_string(index=False))
    print(f"\nSaved Baseline V5 results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


# In[ ]:




