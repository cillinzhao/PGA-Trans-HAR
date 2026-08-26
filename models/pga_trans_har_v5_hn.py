#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations

import argparse
import copy
import json
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

# The user's Anaconda NumPy and PyTorch distributions may bundle separate
# Intel OpenMP runtimes.  Set this before importing either package so the
# script remains runnable in that environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


EPS = 1e-6
MODEL_NAME = "PGA-Trans-HAR-V5"


# ---------------------------------------------------------------------------
# Reproducibility and data
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class PanelData:
    values: np.ndarray
    dates: pd.DatetimeIndex
    names: Tuple[str, ...]
    zero_count: int


def load_panel(path: str | Path, input_format: str = "sqrt_rv") -> PanelData:
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, errors="raise")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate dates are not allowed.")
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    raw = frame.to_numpy(dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 100 or raw.shape[1] < 2:
        raise ValueError(f"Expected a non-empty multivariate panel, got {raw.shape}.")
    finite = raw[np.isfinite(raw)]
    if np.any(finite < 0):
        raise ValueError("Volatility observations cannot be negative.")

    zero_count = int(np.sum(raw == 0.0))
    observed = np.isfinite(raw) & (raw > 0.0)
    values = np.where(observed, raw, np.nan)
    if input_format == "rv":
        values = 100.0 * np.sqrt(values)
    elif input_format != "sqrt_rv":
        raise ValueError("--input-format must be sqrt_rv or rv.")
    return PanelData(
        values=values,
        dates=frame.index,
        names=tuple(map(str, frame.columns)),
        zero_count=zero_count,
    )


def fixed_split(T: int, train_ratio: float, val_ratio: float) -> Tuple[int, int]:
    if not math.isclose(train_ratio, 0.60, abs_tol=1e-12):
        raise ValueError("V5 fixes --train-ratio at 0.60.")
    if not math.isclose(val_ratio, 0.10, abs_tol=1e-12):
        raise ValueError("V5 fixes --val-ratio at 0.10.")
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    if not (22 < train_end < val_end < T):
        raise ValueError("The 60/10/30 split is invalid for this dataset.")
    return train_end, val_end


# ---------------------------------------------------------------------------
# Mask-aware windows and HAR anchor
# ---------------------------------------------------------------------------


def make_windows(
    values: np.ndarray,
    target_start: int,
    target_end: int,
    seq_len: int = 22,
    horizon: int = 1,  # [新增参数]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError("Forecast horizon must be a positive integer.")
    if target_start < 0 or target_end > len(values) or target_start >= target_end:
        raise ValueError("Invalid target range for window construction.")

    xs, x_masks, ys, y_masks, indices = [], [], [], [], []
    for target_idx in range(target_start, target_end):
        input_end = target_idx - horizon + 1    # [修改] 根据 horizon 确定信息截止点
        input_start = input_end - seq_len       # [修改] 
        
        if input_start < 0:                     # [新增] 越界保护
            continue
            
        raw_x = values[input_start : input_end] # [修改]
        raw_y = values[target_idx]
        x_mask = np.isfinite(raw_x).astype(np.float32)
        y_mask = np.isfinite(raw_y).astype(np.float32)
        xs.append(np.nan_to_num(raw_x, nan=0.0).astype(np.float32)[..., None])
        x_masks.append(x_mask[..., None])
        ys.append(np.nan_to_num(raw_y, nan=0.0).astype(np.float32)[..., None])
        y_masks.append(y_mask[..., None])
        indices.append(target_idx)
    return (
        np.asarray(xs),
        np.asarray(x_masks),
        np.asarray(ys),
        np.asarray(y_masks),
        np.asarray(indices, dtype=np.int64),
    )


def numpy_last_valid(x: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # x/mask: [B,S,N]
    B, S, N = x.shape
    reverse = mask[:, ::-1, :].argmax(axis=1)
    position = (S - 1) - reverse
    exists = mask.any(axis=1)
    gathered = np.take_along_axis(x, position[:, None, :], axis=1)[:, 0, :]
    return gathered, exists


def numpy_masked_mean(
    x: np.ndarray, mask: np.ndarray, start: int, end: int
) -> Tuple[np.ndarray, np.ndarray]:
    selected_x = x[:, start:end]
    selected_m = mask[:, start:end]
    counts = selected_m.sum(axis=1)
    means = (selected_x * selected_m).sum(axis=1) / np.maximum(counts, 1)
    return means, counts > 0


def har_features_numpy(
    X: np.ndarray, X_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    values = X[..., 0]
    active = X_mask[..., 0] > 0.5
    S = values.shape[1]
    daily, daily_ok = numpy_last_valid(values, active)
    weekly, weekly_ok = numpy_masked_mean(values, active, max(0, S - 5), S)
    monthly, monthly_ok = numpy_masked_mean(values, active, 0, S)
    features = np.stack([daily, weekly, monthly], axis=-1)
    valid = daily_ok & weekly_ok & monthly_ok
    return features.astype(np.float32), valid


def fit_har_anchor(
    values: np.ndarray, target_end: int, seq_len: int, horizon: int = 1
) -> np.ndarray:
    
    X, Xm, Y, Ym, _ = make_windows(values, seq_len, target_end, seq_len, horizon)
    features, feature_ok = har_features_numpy(X, Xm)
    target = Y[..., 0]
    target_ok = Ym[..., 0] > 0.5
    N = values.shape[1]
    coefficients = np.zeros((N, 4), dtype=np.float64)
    for node in range(N):
        valid = feature_ok[:, node] & target_ok[:, node]
        if valid.sum() < 30:
            raise ValueError(f"Too few HAR observations for node {node}.")
        design = np.column_stack(
            [np.ones(int(valid.sum())), features[valid, node, :]]
        )
        beta, *_ = np.linalg.lstsq(design, target[valid, node], rcond=None)
        coefficients[node] = beta
    return coefficients.astype(np.float32)


def apply_har_numpy(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return (
        coefficients[None, :, 0]
        + np.sum(features * coefficients[None, :, 1:4], axis=-1)
    )


def training_statistics(
    values: np.ndarray, target_end: int, seq_len: int, har_coefficients: np.ndarray, horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    
    X, Xm, Y, Ym, _ = make_windows(values, seq_len, target_end, seq_len, horizon)
    features, _ = har_features_numpy(X, Xm)
    har = np.maximum(apply_har_numpy(features, har_coefficients), EPS)
    y = Y[..., 0]
    mask = Ym[..., 0] > 0.5
    N = y.shape[1]
    scales = np.ones(N, dtype=np.float64)
    har_mse = np.ones(N, dtype=np.float64)
    for node in range(N):
        active = mask[:, node]
        observed = y[active, node]
        scales[node] = max(float(np.std(observed, ddof=1)), 1e-3)
        har_mse[node] = max(
            float(np.mean((har[active, node] - observed) ** 2)), 1e-6
        )
    return scales.astype(np.float32), har_mse.astype(np.float32)


# ---------------------------------------------------------------------------
# Causal rolling ridge-VAR / GFEVD prior
# ---------------------------------------------------------------------------


def causal_fill_window(window: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(window).ffill()
    # Initial closures cannot be forward-filled.  Their fill values use only
    # observations inside the already-observed historical window.
    medians = frame.median(axis=0, skipna=True).fillna(0.0)
    frame = frame.fillna(medians)
    return frame.to_numpy(dtype=np.float64)


def fit_ridge_var(
    data: np.ndarray, lags: int, ridge: float
) -> Tuple[Sequence[np.ndarray], np.ndarray]:
    if lags < 1:
        raise ValueError("VAR lags must be positive.")
    data = np.asarray(data, dtype=np.float64)
    T, N = data.shape
    if T <= lags + 2:
        raise ValueError("Insufficient history for the requested VAR lag.")

    mean = data.mean(axis=0, keepdims=True)
    scale = data.std(axis=0, ddof=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = (data - mean) / scale
    targets = z[lags:]
    lagged = [z[lags - lag : T - lag] for lag in range(1, lags + 1)]
    design = np.column_stack([np.ones(T - lags), *lagged])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ targets
    try:
        coefficients = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(lhs) @ rhs

    residual = targets - design @ coefficients
    sigma = residual.T @ residual / max(len(residual), 1)
    sigma = 0.5 * (sigma + sigma.T) + np.eye(N) * 1e-6
    lag_matrices = []
    for lag in range(lags):
        block = coefficients[1 + lag * N : 1 + (lag + 1) * N]
        # Regression stores source-by-target; VAR convention is target-by-source.
        lag_matrices.append(block.T)
    return lag_matrices, sigma


def ridge_var_gfevd(
    history: np.ndarray,
    lags: int,
    ridge: float,
    horizon: int,
) -> np.ndarray:
    complete = causal_fill_window(history)
    lag_matrices, sigma = fit_ridge_var(complete, lags=lags, ridge=ridge)
    N = complete.shape[1]
    psi = [np.eye(N, dtype=np.float64)]
    for h in range(1, horizon):
        current = np.zeros((N, N), dtype=np.float64)
        for lag, A_lag in enumerate(lag_matrices, start=1):
            if lag <= h:
                current += A_lag @ psi[h - lag]
        psi.append(current)

    theta = np.zeros((N, N), dtype=np.float64)
    sigma_diag = np.maximum(np.diag(sigma), 1e-10)
    for target in range(N):
        denominator = 0.0
        for phi in psi:
            denominator += float((phi @ sigma @ phi.T)[target, target])
        denominator = max(denominator, 1e-10)
        for source in range(N):
            numerator = 0.0
            sigma_column = sigma[:, source]
            for phi in psi:
                effect = float((phi @ sigma_column)[target])
                numerator += effect * effect
            theta[target, source] = numerator / sigma_diag[source] / denominator

    theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)
    theta = np.maximum(theta, 0.0)
    row_sum = theta.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= 1e-12
    theta[empty] = np.eye(N)[empty]
    theta /= np.maximum(theta.sum(axis=1, keepdims=True), 1e-12)
    return theta.astype(np.float32)


def build_dynamic_priors(
    values: np.ndarray,
    target_indices: Iterable[int],
    seq_len: int,
    graph_window: int,
    graph_update_every: int,
    var_lag: int,
    ridge: float,
    dy_horizon: int,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if graph_window < seq_len:
        raise ValueError("--graph-window must be at least seq_len.")
    if graph_update_every < 1:
        raise ValueError("--graph-update-every must be positive.")
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be a positive integer.")
    cache: Dict[int, np.ndarray] = {}
    diagnostics = []
    output = []
    for target_idx in map(int, target_indices):
        # target_idx is the date being forecast.  The latest observable input
        # is target_idx - forecast_horizon, hence origin_end is the exclusive
        # history endpoint.  Align the update schedule on this origin, not on
        # the future target date.  This preserves the requested update interval
        # for every forecast horizon and prevents the h=5/22 schedules from
        # degenerating into near-daily graph re-estimation.
        origin_end = target_idx - forecast_horizon + 1
        if origin_end < seq_len:
            raise ValueError(
                f"Target {target_idx} has insufficient history for "
                f"h={forecast_horizon} and seq_len={seq_len}."
            )
        graph_end = seq_len + (
            (origin_end - seq_len) // graph_update_every
        ) * graph_update_every
        if graph_end > origin_end:
            raise AssertionError("Dynamic graph endpoint exceeds forecast origin.")
        if graph_end not in cache:
            graph_start = max(0, graph_end - graph_window)
            history = values[graph_start:graph_end]
            graph = ridge_var_gfevd(
                history,
                lags=var_lag,
                ridge=ridge,
                horizon=dy_horizon,
            )
            cache[graph_end] = graph
            diagnostics.append(
                {
                    "graph_end_exclusive": graph_end,
                    "graph_start": graph_start,
                    "forecast_horizon": int(forecast_horizon),
                    "graph_update_every": int(graph_update_every),
                    "n_union_days": graph_end - graph_start,
                    "min_active_per_node": int(
                        np.isfinite(history).sum(axis=0).min()
                    ),
                    "max_row_sum_error": float(
                        np.max(np.abs(graph.sum(axis=1) - 1.0))
                    ),
                }
            )
        output.append(cache[graph_end])
    return np.asarray(output, dtype=np.float32), pd.DataFrame(diagnostics)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ForecastDataset(Dataset):
    def __init__(self, window_tuple, priors: np.ndarray):
        X, Xm, Y, Ym, indices = window_tuple
        if len(X) != len(priors):
            raise ValueError("One dynamic prior is required for every window.")
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Xm = torch.as_tensor(Xm, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.Ym = torch.as_tensor(Ym, dtype=torch.float32)
        self.priors = torch.as_tensor(priors, dtype=torch.float32)
        self.indices = torch.as_tensor(indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return (
            self.X[index],
            self.Xm[index],
            self.Y[index],
            self.Ym[index],
            self.priors[index],
            self.indices[index],
        )


def make_loader(dataset: ForecastDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


# ---------------------------------------------------------------------------
# PGA model
# ---------------------------------------------------------------------------


def masked_softmax(
    scores: torch.Tensor, key_mask: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    active = key_mask.to(dtype=torch.bool)
    masked = scores.masked_fill(~active, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked, dim=dim)
    weights = torch.where(active, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=dim, keepdim=True)
    return weights / denominator.clamp_min(1e-12)


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        B, S, N, H = x.shape
        sequence = x.transpose(1, 2).reshape(B * N, S, H)
        active = x_mask[..., 0].transpose(1, 2).reshape(B * N, S)
        q = self.q(sequence).view(B * N, S, self.num_heads, self.head_dim)
        k = self.k(sequence).view(B * N, S, self.num_heads, self.head_dim)
        v = self.v(sequence).view(B * N, S, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        key_mask = active[:, None, None, :].expand(-1, self.num_heads, S, -1)
        attention = self.dropout(masked_softmax(scores, key_mask))
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(B * N, S, H)
        output = self.out(output).reshape(B, N, S, H).transpose(1, 2)
        return output


class DynamicPriorGuidedSpatialAttention(nn.Module):
    """Spatial attention with g=1 for data attention and g=0 for prior."""

    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        seq_len: int,
        dropout: float,
        node_emb_dim: int = 8,
        time_emb_dim: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(seq_len, time_emb_dim) * 0.02)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + node_emb_dim + time_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_intercept = nn.Parameter(torch.full((num_nodes, 1), -0.5))
        self.attention_dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        prior: torch.Tensor,
        return_gate: bool = False,
    ):
        B, S, N, H = x.shape
        q = self.q(x).reshape(B * S, N, H)
        k = self.k(x).reshape(B * S, N, H)
        v = self.v(x).reshape(B * S, N, H)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale

        source_active = x_mask[..., 0].reshape(B * S, N)
        key_mask = source_active[:, None, :].expand(-1, N, -1)
        data_attention = masked_softmax(scores, key_mask)

        prior_expanded = prior[:, None, :, :].expand(B, S, N, N)
        prior_expanded = prior_expanded.reshape(B * S, N, N)
        masked_prior = prior_expanded * key_mask.to(prior_expanded.dtype)
        prior_sum = masked_prior.sum(dim=-1, keepdim=True)
        # A union trading day should have at least one active source.  The
        # fallback remains defensive for other datasets.
        normalized_prior = masked_prior / prior_sum.clamp_min(1e-12)
        normalized_prior = torch.where(
            prior_sum > 1e-12, normalized_prior, data_attention
        )

        node_emb = self.node_embedding[None, None, :, :].expand(B, S, -1, -1)
        time_emb = self.time_embedding[None, :, None, :].expand(B, -1, N, -1)
        gate_input = torch.cat([x, node_emb, time_emb], dim=-1)
        gate = torch.sigmoid(
            self.gate_net(gate_input)
            + self.gate_intercept[None, None, :, :]
        ).reshape(B * S, N, 1)

        hybrid = gate * data_attention + (1.0 - gate) * normalized_prior
        hybrid = self.attention_dropout(hybrid)
        output = torch.bmm(hybrid, v).reshape(B, S, N, H)
        output = self.out(output)
        gate = gate.reshape(B, S, N)
        return output, (gate if return_gate else None)


class SpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_nodes: int,
        seq_len: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.temporal = MultiHeadTemporalAttention(hidden_dim, num_heads, dropout)
        self.spatial = DynamicPriorGuidedSpatialAttention(
            hidden_dim, num_nodes, seq_len, dropout
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
        # Inactive nodes are allowed to receive information.  They are excluded
        # only as sources/keys, not destroyed as target/query states.
        temporal = self.temporal(x, x_mask)
        x = self.norm1(x + self.dropout(temporal))
        spatial, gate = self.spatial(x, x_mask, prior, return_gate=return_gate)
        x = self.norm2(x + self.dropout(spatial))
        x = self.norm3(x + self.ffn(x))
        return x, gate


def torch_last_valid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, S, N, D = x.shape
    active = mask[..., 0] > 0.5
    reverse = active.flip(dims=[1]).to(torch.int64).argmax(dim=1)
    position = (S - 1) - reverse
    exists = active.any(dim=1)
    gathered = torch.gather(
        x, 1, position[:, None, :, None].expand(-1, 1, -1, D)
    ).squeeze(1)
    return gathered * exists[..., None]


def torch_masked_mean(
    x: torch.Tensor, mask: torch.Tensor, start: int, end: int
) -> torch.Tensor:
    selected_x = x[:, start:end]
    selected_m = mask[:, start:end]
    return (selected_x * selected_m).sum(dim=1) / selected_m.sum(dim=1).clamp_min(1.0)


def inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    y = y.clamp_min(EPS)
    return y + torch.log(-torch.expm1(-y))


class PGATransHARV5(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        har_coefficients: np.ndarray,
        seq_len: int = 22,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.max_residual_scale = float(max_residual_scale)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, seq_len, 1, hidden_dim) * 0.02
        )
        self.node_embedding = nn.Parameter(
            torch.randn(1, 1, num_nodes, hidden_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            [
                SpatioTemporalBlock(
                    hidden_dim, num_nodes, seq_len, num_heads, dropout
                )
                for _ in range(num_blocks)
            ]
        )
        residual_hidden = max(hidden_dim // 2, 8)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, residual_hidden),
            nn.GELU(),
            nn.Linear(residual_hidden, 1),
        )
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_dim + 2, residual_hidden),
            nn.GELU(),
            nn.Linear(residual_hidden, 1),
        )
        nn.init.constant_(self.residual_gate[-1].bias, -1.5)
        # alpha=0 makes the initial prediction exactly equal to HAR while the
        # randomly initialised residual head supplies a non-zero alpha gradient.
        self.alpha_raw = nn.Parameter(torch.zeros(num_nodes, 1))
        coefficients = np.asarray(har_coefficients, dtype=np.float32)
        if coefficients.shape != (num_nodes, 4):
            raise ValueError("HAR coefficient shape mismatch.")
        self.register_buffer("har_coefficients", torch.as_tensor(coefficients))

    def har_anchor(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        daily = torch_last_valid(x, x_mask)
        weekly = torch_masked_mean(x, x_mask, self.seq_len - 5, self.seq_len)
        monthly = torch_masked_mean(x, x_mask, 0, self.seq_len)
        features = torch.cat([daily, weekly, monthly], dim=-1)
        raw = (
            self.har_coefficients[None, :, 0:1]
            + (features * self.har_coefficients[None, :, 1:4]).sum(
                dim=-1, keepdim=True
            )
        )
        return raw.clamp_min(EPS)

    def forward(self, x, x_mask, prior, return_aux=False):
        model_input = torch.cat([x, x_mask], dim=-1)
        hidden = (
            self.input_projection(model_input)
            + self.position_embedding
            + self.node_embedding
        )
        gates = []
        for block in self.blocks:
            hidden, gate = block(
                hidden, x_mask, prior, return_gate=return_aux
            )
            if return_aux:
                gates.append(gate)

        # Calendar-end state includes information received on closure days.
        last_hidden = hidden[:, -1]
        active_rate = x_mask[..., 0].mean(dim=1, keepdim=False)[..., None]
        last_day_active = x_mask[:, -1]
        residual_gate = torch.sigmoid(
            self.residual_gate(
                torch.cat([last_hidden, active_rate, last_day_active], dim=-1)
            )
        )
        residual = torch.tanh(self.residual_head(last_hidden))
        alpha = self.max_residual_scale * torch.tanh(self.alpha_raw)[None, :, :]

        anchor = self.har_anchor(x, x_mask)
        latent = inverse_softplus(anchor) + alpha * residual_gate * residual
        prediction = F.softplus(latent).clamp_min(EPS)
        if not return_aux:
            return prediction
        return prediction, {
            "attention_gate": torch.stack(gates, dim=0),
            "residual_gate": residual_gate,
            "alpha": alpha,
            "har_anchor": anchor,
            "residual": residual,
        }


# ---------------------------------------------------------------------------
# Objective, training and prediction
# ---------------------------------------------------------------------------


class RobustRelativeMSE(nn.Module):
    def __init__(
        self,
        node_scale: np.ndarray,
        har_reference_mse: np.ndarray,
        robust_lambda: float,
        temperature: float,
    ):
        super().__init__()
        self.register_buffer("node_scale", torch.as_tensor(node_scale).view(1, -1))
        self.register_buffer(
            "har_reference_mse", torch.as_tensor(har_reference_mse).view(1, -1)
        )
        self.robust_lambda = float(robust_lambda)
        self.temperature = float(temperature)

    def components(self, prediction, target, mask):
        error2 = (prediction[..., 0] - target[..., 0]) ** 2
        active = mask[..., 0]
        counts = active.sum(dim=0)
        valid = counts > 0
        node_mse = (error2 * active).sum(dim=0) / counts.clamp_min(1.0)
        standardized = node_mse / self.node_scale[0].square().clamp_min(1e-8)
        base = standardized[valid].mean()
        relative = node_mse / self.har_reference_mse[0].clamp_min(1e-8)
        relative_valid = relative[valid]
        centered = relative_valid - 1.0
        smooth_worst = self.temperature * torch.logsumexp(
            centered / self.temperature, dim=0
        ) - self.temperature * math.log(max(int(valid.sum().item()), 1))
        total = base + self.robust_lambda * smooth_worst
        return total, base, smooth_worst, node_mse, relative

    def forward(self, prediction, target, mask):
        return self.components(prediction, target, mask)[0]


@dataclass
class TrainConfig:
    seq_len: int = 22
    hidden_dim: int = 32
    num_blocks: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 20
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_residual_scale: float = 1.0
    robust_lambda: float = 0.10
    robust_temperature: float = 0.20


def move_batch(batch, device: torch.device):
    x, xm, y, ym, prior, index = batch
    return (
        x.to(device),
        xm.to(device),
        y.to(device),
        ym.to(device),
        prior.to(device),
        index,
    )


def train_epoch(model, loader, objective, optimizer, config, device) -> float:
    model.train()
    total, batches = 0.0, 0
    for batch in loader:
        x, xm, y, ym, prior, _ = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, xm, prior)
        loss = objective(prediction, y, ym)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def collect_predictions(model, loader, device, return_aux=False):
    model.eval()
    predictions, targets, masks, indices = [], [], [], []
    attention_gates, residual_gates, anchors, residuals = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x, xm, y, ym, prior, index = move_batch(batch, device)
            if return_aux:
                prediction, aux = model(x, xm, prior, return_aux=True)
                attention_gates.append(aux["attention_gate"].cpu().numpy())
                residual_gates.append(aux["residual_gate"].cpu().numpy())
                anchors.append(aux["har_anchor"].cpu().numpy())
                residuals.append(aux["residual"].cpu().numpy())
            else:
                prediction = model(x, xm, prior)
            predictions.append(prediction.cpu().numpy()[..., 0])
            targets.append(y.cpu().numpy()[..., 0])
            masks.append(ym.cpu().numpy()[..., 0])
            indices.append(index.numpy())
    output = {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "mask": np.concatenate(masks),
        "target_idx": np.concatenate(indices),
    }
    if return_aux:
        # attention batches are [blocks,B,S,N], so concatenate along B.
        output["attention_gate"] = np.concatenate(attention_gates, axis=1)
        output["residual_gate"] = np.concatenate(residual_gates)[..., 0]
        output["har_anchor"] = np.concatenate(anchors)[..., 0]
        output["residual"] = np.concatenate(residuals)[..., 0]
    return output


def objective_from_arrays(
    arrays: dict,
    node_scale: np.ndarray,
    har_reference_mse: np.ndarray,
    robust_lambda: float,
    temperature: float,
) -> dict:
    prediction = arrays["prediction"]
    target = arrays["target"]
    mask = arrays["mask"] > 0.5
    node_mse = np.zeros(target.shape[1], dtype=np.float64)
    for node in range(target.shape[1]):
        active = mask[:, node]
        node_mse[node] = np.mean(
            (prediction[active, node] - target[active, node]) ** 2
        )
    base = float(np.mean(node_mse / np.maximum(node_scale, 1e-8) ** 2))
    relative = node_mse / np.maximum(har_reference_mse, 1e-8)
    centered = (relative - 1.0) / temperature
    maximum = float(np.max(centered))
    log_mean_exp = maximum + math.log(float(np.mean(np.exp(centered - maximum))))
    smooth_worst = float(temperature * log_mean_exp)
    return {
        "objective": base + robust_lambda * smooth_worst,
        "base_standardized_mse": base,
        "smooth_worst_relative_regret": smooth_worst,
        "mean_raw_mse": float(node_mse.mean()),
        "max_relative_mse": float(relative.max()),
    }


def build_model(
    num_nodes: int,
    har_coefficients: np.ndarray,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> PGATransHARV5:
    set_seed(seed)
    model = PGATransHARV5(
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


def select_epoch_on_validation(
    model,
    train_loader,
    val_loader,
    objective,
    node_scale,
    har_reference_mse,
    config,
    device,
):
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.max_epochs, 1), eta_min=config.lr_min
    )
    best_epoch, best_value, wait = 0, float("inf"), 0
    best_state = None
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, objective, optimizer, config, device
        )
        validation = collect_predictions(model, val_loader, device)
        scores = objective_from_arrays(
            validation,
            node_scale,
            har_reference_mse,
            config.robust_lambda,
            config.robust_temperature,
        )
        history.append(
            {
                "epoch": epoch,
                "train_objective": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"val_{key}": value for key, value in scores.items()},
            }
        )
        if scores["objective"] < best_value - 1e-8:
            best_value = scores["objective"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        scheduler.step()
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Validation failed to produce a checkpoint.")
    model.load_state_dict(best_state)
    return model, best_epoch, pd.DataFrame(history)


def refit_fixed_epochs(
    model,
    loader,
    objective,
    epochs,
    config,
    device,
):
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=config.lr_min
    )
    final_loss = float("nan")
    for _ in range(epochs):
        final_loss = train_epoch(
            model, loader, objective, optimizer, config, device
        )
        scheduler.step()
    return model, final_loss


# ---------------------------------------------------------------------------
# Metrics and exports
# ---------------------------------------------------------------------------


def metric_table(target, prediction, mask, names, seed, horizon) -> pd.DataFrame:
    rows = []
    mask = mask > 0.5
    for node, name in enumerate(names):
        active = mask[:, node]
        error = prediction[active, node] - target[active, node]
        rows.append(
            {
                "model": MODEL_NAME,
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


def prediction_frame(arrays, dates, names, seed, horizon) -> pd.DataFrame:
    rows = []
    for row, target_idx in enumerate(arrays["target_idx"]):
        for node, name in enumerate(names):
            active = bool(arrays["mask"][row, node] > 0.5)
            rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "target_idx": int(target_idx),
                    "target_date": pd.Timestamp(dates[target_idx]).date().isoformat(),
                    "origin_idx": int(target_idx - horizon),
                    "origin_date": pd.Timestamp(dates[target_idx - horizon]).date().isoformat(),
                    "node": node,
                    "node_name": name,
                    "y_sqrt_rv_pct": (
                        float(arrays["target"][row, node]) if active else np.nan
                    ),
                    "pred_sqrt_rv_pct": float(arrays["prediction"][row, node]),
                    "active": int(active),
                }
            )
    return pd.DataFrame(rows)


def gate_frame(arrays, dates, names, seed, alpha, horizon) -> pd.DataFrame:
    attention = arrays["attention_gate"]
    # Last block, final calendar input position; also retain window average.
    last_gate = attention[-1, :, -1, :]
    window_gate = attention[-1].mean(axis=1)
    rows = []
    for row, target_idx in enumerate(arrays["target_idx"]):
        for node, name in enumerate(names):
            rows.append(
                {
                    "model": MODEL_NAME,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "target_idx": int(target_idx),
                    "target_date": pd.Timestamp(dates[target_idx]).date().isoformat(),
                    "node": node,
                    "node_name": name,
                    "attention_gate_g": float(last_gate[row, node]),
                    "attention_gate_g_window_mean": float(window_gate[row, node]),
                    "residual_gate_r": float(arrays["residual_gate"][row, node]),
                    "node_residual_scale_alpha": float(alpha[node]),
                    "har_anchor": float(arrays["har_anchor"][row, node]),
                    "raw_bounded_residual": float(arrays["residual"][row, node]),
                    "target_active": int(arrays["mask"][row, node] > 0.5),
                }
            )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            ["model", "horizon", "node", "node_name"], as_index=False
        )
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


def ensemble_outputs(predictions: pd.DataFrame, horizon: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = [
        "horizon", "target_idx", "target_date", "origin_idx", "origin_date",
        "node", "node_name",
    ]
    ensemble = (
        predictions.groupby(keys, as_index=False)
        .agg(
            y_sqrt_rv_pct=("y_sqrt_rv_pct", "mean"),
            pred_sqrt_rv_pct=("pred_sqrt_rv_pct", "mean"),
            pred_seed_std=("pred_sqrt_rv_pct", "std"),
            active=("active", "min"),
            n_seeds=("seed", "nunique"),
        )
    )
    ensemble.insert(0, "seed", -1)
    ensemble.insert(0, "model", f"{MODEL_NAME}-SeedEnsemble")
    ensemble["pred_seed_std"] = ensemble["pred_seed_std"].fillna(0.0)
    return ensemble, metrics_from_ensemble(ensemble, horizon)


def metrics_from_ensemble(ensemble: pd.DataFrame, horizon: int) -> pd.DataFrame:
    metric_rows = []
    for (node, name), group in ensemble.groupby(["node", "node_name"]):
        active = group["active"].to_numpy(dtype=bool)
        y = group.loc[active, "y_sqrt_rv_pct"].to_numpy(dtype=float)
        p = group.loc[active, "pred_sqrt_rv_pct"].to_numpy(dtype=float)
        metric_rows.append(
            {
                "model": f"{MODEL_NAME}-SeedEnsemble",
                "horizon": int(horizon),
                "node": int(node),
                "node_name": name,
                "n_active": int(active.sum()),
                "MSE": float(np.mean((p - y) ** 2)),
                "MAE": float(np.mean(np.abs(p - y))),
                "n_seeds": int(group["n_seeds"].min()),
            }
        )
    return pd.DataFrame(metric_rows)


def date_manifests(test_windows, dates, names, horizon) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _, _, _, target_mask, indices = test_windows
    rows = []
    for row, target_idx in enumerate(indices):
        common = bool((target_mask[row, :, 0] > 0.5).all())
        for node, name in enumerate(names):
            rows.append(
                {
                    "horizon": int(horizon),
                    "target_idx": int(target_idx),
                    "target_date": pd.Timestamp(dates[target_idx]).date().isoformat(),
                    "origin_idx": int(target_idx - horizon),
                    "origin_date": pd.Timestamp(dates[target_idx - horizon]).date().isoformat(),
                    "node": node,
                    "node_name": name,
                    "active": int(target_mask[row, node, 0] > 0.5),
                    "all_markets_active": int(common),
                }
            )
    all_days = pd.DataFrame(rows)
    common_days = all_days[all_days["all_markets_active"] == 1].copy()
    return all_days, common_days


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="df_union_sqrt.csv", help="CSV path, e.g. df_union_sqrt.csv")
    parser.add_argument("--input-format", choices=["sqrt_rv", "rv"], default="sqrt_rv")
    parser.add_argument("--outdir", default="pga_results_v5")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 22])  # 新增 horizons 参数
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
    parser.add_argument("--graph-window", type=int, default=252)
    parser.add_argument("--graph-update-every", type=int, default=20)
    parser.add_argument("--var-lag", type=int, default=2)
    parser.add_argument("--var-ridge", type=float, default=5.0)
    parser.add_argument("--dy-horizon", type=int, default=22)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--no-refit",
        action="store_true",
        help="Use the best 60%%-training checkpoint directly instead of refitting on 70%%.",
    )
    args, unknown = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    args.horizons = sorted(set(args.horizons))
    if not args.horizons or any(horizon < 1 for horizon in args.horizons):
        raise ValueError("--horizons must contain positive integers.")
    if args.seq_len != 22:
        raise ValueError("V5 fixes the HAR lookback at 22 days.")
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Epoch and patience values must be positive.")
    if args.robust_temperature <= 0 or args.robust_lambda < 0:
        raise ValueError("Invalid robust-loss settings.")

    panel = load_panel(args.data, args.input_format)
    values, dates, names = panel.values, panel.dates, panel.names
    T, N = values.shape
    train_end, val_end = fixed_split(T, args.train_ratio, args.val_ratio)
    device = choose_device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
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

    # 用于收集所有 horizon 最终的整合数据
    global_summary_metrics = []
    global_ensemble_metrics = []
    global_ensemble_common_metrics = []
    parameter_count = None

    # ---- 核心：新增外层 Horizon 遍历 ----
    for h in args.horizons:
        print(f"\n{'='*60}")
        print(f"=== Starting Forecast Horizon: h = {h} ===")
        print(f"{'='*60}\n")
        
        # 为每个 horizon 创建独立的输出目录
        outdir_h = outdir / f"h{h}"
        outdir_h.mkdir(parents=True, exist_ok=True)

        # The first validation/test target must have an origin on or after the
        # preceding split endpoint.  Starting directly at train_end/val_end
        # would let an h-step model use labels dated after the first forecast
        # origin when h > 1.
        validation_target_start = train_end + h - 1
        test_target_start = val_end + h - 1

        # 构建支持 horizon 的 Windows
        train_windows = make_windows(values, args.seq_len, train_end, args.seq_len, horizon=h)
        val_windows = make_windows(
            values, validation_target_start, val_end, args.seq_len, horizon=h
        )
        refit_windows = make_windows(values, args.seq_len, val_end, args.seq_len, horizon=h)
        test_windows = make_windows(
            values, test_target_start, T, args.seq_len, horizon=h
        )

        if int(val_windows[4][0]) - h != train_end - 1:
            raise AssertionError("Validation forecast origin is not split-safe.")
        if int(test_windows[4][0]) - h != val_end - 1:
            raise AssertionError("Test forecast origin is not split-safe.")
        
        # 获取所有生成过的合法 target index，避免遗漏
        all_target_indices = np.unique(np.concatenate([
            train_windows[4], val_windows[4], refit_windows[4], test_windows[4]
        ]))

        all_priors, graph_diagnostics = build_dynamic_priors(
            values,
            all_target_indices,
            seq_len=args.seq_len,
            graph_window=args.graph_window,
            graph_update_every=args.graph_update_every,
            var_lag=args.var_lag,
            ridge=args.var_ridge,
            dy_horizon=args.dy_horizon,
            forecast_horizon=h
        )
        prior_by_target = {
            int(index): prior for index, prior in zip(all_target_indices, all_priors)
        }

        def priors_for(window_tuple):
            return np.asarray(
                [prior_by_target[int(index)] for index in window_tuple[-1]],
                dtype=np.float32,
            )

        train_dataset = ForecastDataset(train_windows, priors_for(train_windows))
        val_dataset = ForecastDataset(val_windows, priors_for(val_windows))
        refit_dataset = ForecastDataset(refit_windows, priors_for(refit_windows))
        test_dataset = ForecastDataset(test_windows, priors_for(test_windows))
        
        val_loader = make_loader(val_dataset, config.batch_size, shuffle=False)
        test_loader = make_loader(test_dataset, config.batch_size, shuffle=False)

        # 计算支持 horizon 的 HAR Baseline
        har_dev = fit_har_anchor(values, train_end, args.seq_len, horizon=h)
        scale_dev, har_mse_dev = training_statistics(
            values, train_end, args.seq_len, har_dev, horizon=h
        )
        har_refit = fit_har_anchor(values, val_end, args.seq_len, horizon=h)
        scale_refit, har_mse_refit = training_statistics(
            values, val_end, args.seq_len, har_refit, horizon=h
        )

        graph_diagnostics["graph_end_date"] = graph_diagnostics[
            "graph_end_exclusive"
        ].map(lambda end: pd.Timestamp(dates[int(end) - 1]).date().isoformat())
        graph_diagnostics.to_csv(outdir_h / f"dynamic_graph_diagnostics_h{h}.csv", index=False)
        
        all_manifest, common_manifest = date_manifests(test_windows, dates, names, horizon=h)
        all_manifest.to_csv(outdir_h / f"test_date_manifest_all_days_h{h}.csv", index=False)
        common_manifest.to_csv(outdir_h / f"test_date_manifest_common_days_h{h}.csv", index=False)

        all_metrics, all_predictions, all_gates, histories = [], [], [], []
        
        for seed in args.seeds:
            print(f"\n--- [h={h}] Seed {seed}: development training and validation ---", flush=True)
            train_loader = make_loader(train_dataset, config.batch_size, shuffle=True)
            dev_objective = RobustRelativeMSE(
                scale_dev,
                har_mse_dev,
                config.robust_lambda,
                config.robust_temperature,
            ).to(device)
            
            dev_model = build_model(N, har_dev, config, seed, device)
            if parameter_count is None:
                parameter_count = sum(
                    parameter.numel()
                    for parameter in dev_model.parameters()
                    if parameter.requires_grad
                )
                
            dev_model, best_epoch, history = select_epoch_on_validation(
                dev_model,
                train_loader,
                val_loader,
                dev_objective,
                scale_dev,
                har_mse_dev,
                config,
                device,
            )
            history.insert(0, "seed", seed)
            history["selected_epoch"] = best_epoch
            histories.append(history)
            print(f"Selected epoch: {best_epoch}", flush=True)

            if args.no_refit:
                final_model = dev_model
                final_train_loss = float("nan")
                final_alpha_reference = har_dev
            else:
                print(
                    f"Refitting from scratch on the first 70% for {best_epoch} epochs...",
                    flush=True,
                )
                refit_loader = make_loader(refit_dataset, config.batch_size, shuffle=True)
                refit_objective = RobustRelativeMSE(
                    scale_refit,
                    har_mse_refit,
                    config.robust_lambda,
                    config.robust_temperature,
                ).to(device)
                
                final_model = build_model(N, har_refit, config, seed, device)
                final_model, final_train_loss = refit_fixed_epochs(
                    final_model,
                    refit_loader,
                    refit_objective,
                    best_epoch,
                    config,
                    device,
                )
                final_alpha_reference = har_refit

            # 评估测试集
            test_arrays = collect_predictions(
                final_model, test_loader, device, return_aux=True
            )
            metrics = metric_table(
                test_arrays["target"],
                test_arrays["prediction"],
                test_arrays["mask"],
                names,
                seed,
                horizon=h,
            )
            metrics["selected_epoch"] = best_epoch
            metrics["final_train_objective"] = final_train_loss
            predictions = prediction_frame(test_arrays, dates, names, seed, horizon=h)
            alpha = (
                config.max_residual_scale
                * torch.tanh(final_model.alpha_raw.detach())
            ).cpu().numpy()[:, 0]
            gates = gate_frame(test_arrays, dates, names, seed, alpha, horizon=h)

            seed_dir = outdir_h / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(seed_dir / f"metrics_h{h}_seed{seed}.csv", index=False)
            predictions.to_csv(
                seed_dir / f"{MODEL_NAME}_h{h}_seed{seed}_predictions.csv", index=False
            )
            gates.to_csv(seed_dir / f"gates_h{h}_seed{seed}.csv", index=False)
            
            torch.save(
                {
                    "model_state_dict": final_model.state_dict(),
                    "seed": seed,
                    "selected_epoch": best_epoch,
                    "train_config": asdict(config),
                    "har_coefficients": final_alpha_reference,
                },
                seed_dir / f"model_h{h}_seed{seed}.pt",
            )
            
            all_metrics.append(metrics)
            all_predictions.append(predictions)
            all_gates.append(gates)

        # --- 整合当前 Horizon 的结果 ---
        metrics_frame = pd.concat(all_metrics, ignore_index=True)
        prediction_frame_all = pd.concat(all_predictions, ignore_index=True)
        gate_frame_all = pd.concat(all_gates, ignore_index=True)
        history_frame = pd.concat(histories, ignore_index=True)
        
        summary = summarize_metrics(metrics_frame)
        ensemble, ensemble_metrics = ensemble_outputs(prediction_frame_all, horizon=h)
        
        common_target_indices = set(common_manifest["target_idx"].astype(int).unique())
        ensemble_common = ensemble[
            ensemble["target_idx"].astype(int).isin(common_target_indices)
        ].copy()
        ensemble_common_metrics = metrics_from_ensemble(ensemble_common, horizon=h)

        # 存到当前 h 的目录里
        metrics_frame.to_csv(outdir_h / f"all_seed_metrics_h{h}.csv", index=False)
        summary.to_csv(outdir_h / f"summary_mean_std_h{h}.csv", index=False)
        prediction_frame_all.to_csv(outdir_h / f"all_seed_predictions_h{h}.csv", index=False)
        gate_frame_all.to_csv(outdir_h / f"all_seed_gates_h{h}.csv", index=False)
        history_frame.to_csv(outdir_h / f"validation_history_h{h}.csv", index=False)
        ensemble.to_csv(outdir_h / f"seed_ensemble_predictions_h{h}.csv", index=False)
        ensemble_metrics.to_csv(outdir_h / f"seed_ensemble_metrics_h{h}.csv", index=False)
        ensemble_common.to_csv(
            outdir_h / f"seed_ensemble_predictions_common_days_h{h}.csv", index=False
        )
        ensemble_common_metrics.to_csv(
            outdir_h / f"seed_ensemble_metrics_common_days_h{h}.csv", index=False
        )
        
        # 加入全局列表，用于最后在终端打印和输出总的汇总 csv
        global_summary_metrics.append(summary)
        global_ensemble_metrics.append(ensemble_metrics)
        global_ensemble_common_metrics.append(ensemble_common_metrics)

    # ==== 循环结束，在根目录输出所有 horizon 整体的结果及 JSON 配置 ====
    final_summary = pd.concat(global_summary_metrics, ignore_index=True)
    final_ensemble = pd.concat(global_ensemble_metrics, ignore_index=True)
    final_ensemble_common = pd.concat(global_ensemble_common_metrics, ignore_index=True)
    
    final_summary.to_csv(outdir / "summary_mean_std_ALL_HORIZONS.csv", index=False)
    final_ensemble.to_csv(outdir / "seed_ensemble_metrics_ALL_HORIZONS.csv", index=False)
    final_ensemble_common.to_csv(outdir / "seed_ensemble_metrics_common_days_ALL_HORIZONS.csv", index=False)

    with open(outdir / "run_config.json", "w", encoding="utf-8") as stream:
        json.dump(
            {
                "model": MODEL_NAME,
                "data": str(args.data),
                "input_format": args.input_format,
                "zero_codes_restored_to_nan": panel.zero_count,
                "rows": T,
                "nodes": N,
                "horizons": args.horizons,
                "train_end_exclusive": train_end,
                "validation_end_exclusive": val_end,
                "train_last_date": dates[train_end - 1].date().isoformat(),
                "validation_split_first_date": dates[train_end].date().isoformat(),
                "validation_last_date": dates[val_end - 1].date().isoformat(),
                "validation_first_target_date_by_horizon": {
                    str(h): dates[train_end + h - 1].date().isoformat()
                    for h in args.horizons
                },
                "test_first_target_date_by_horizon": {
                    str(h): dates[val_end + h - 1].date().isoformat()
                    for h in args.horizons
                },
                "test_split_first_date": dates[val_end].date().isoformat(),
                "test_last_date": dates[-1].date().isoformat(),
                "split_protocol": "60_train_10_validation_30_test",
                "refit_on_first_70_percent": not args.no_refit,
                "test_used_for_selection": False,
                "seeds": args.seeds,
                "device": str(device),
                "parameter_count": parameter_count,
                "train_config": asdict(config),
                "dynamic_graph": {
                    "method": "causal_rolling_ridge_VAR_GFEVD",
                    "window": args.graph_window,
                    "update_every": args.graph_update_every,
                    "var_lag": args.var_lag,
                    "ridge": args.var_ridge,
                    "dy_horizon": args.dy_horizon,
                },
                "residual_parameterization": (
                    "softplus(inv_softplus(HAR_anchor) + "
                    "alpha_node * residual_gate_r * tanh(residual))"
                ),
                "attention_gate_interpretation": "g=1 data attention; g=0 prior graph",
                "training_objective": (
                    "mean node-standardized MSE + lambda * "
                    "smooth-worst(node MSE / training HAR MSE - 1)"
                ),
                "reported_metrics": ["MSE", "MAE"],
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "="*60)
    print("ALL HORIZONS DONE")
    print("="*60)
    print("\nSeed mean/std metrics")
    print(final_summary.to_string(index=False))
    print("\nSeed-ensemble metrics")
    print(final_ensemble.to_string(index=False))
    print("\nSeed-ensemble metrics on common trading days")
    print(final_ensemble_common.to_string(index=False))
    print(f"\nSaved complete V5 results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()


# In[ ]:


