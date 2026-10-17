#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aligned MSE-loss retest baselines for the eight-market volatility panel.

HAR, VHAR and HAR-KS remain OLS on volatility. GNN-HAR, DCRNN-HAR and
Pure-ST-Transformer minimize training-scale-standardized, node-balanced MSE
on volatility. They share the union-calendar targets and chronological
evaluation protocol of the main model; QLIKE is a reported metric only.

DCRNN-HAR follows Chi, Gao and Wang (2026, Journal of Forecasting,
doi:10.1002/for.70081): a sample-specific 22-day GFEVD graph, daily
source masks, DCGRU encoder AND autoregressive decoder, nonoverlapping
learnable HAR head, and horizon-length output. No teacher forcing and no
future realized volatility enter the decoder. Future market-activity flags
are used as a *known-calendar* input, as in that paper; this assumes closures
are announced by the forecast origin. They are never used as volatility
values. If unscheduled closures exist, this assumption needs separate audit.

Run e.g.:
 python baselines/baseline_v5_hn.py
 python baselines/baseline_v5_hn.py --models HAR VHAR HAR-KS --horizons 1 5 22
 python baselines/baseline_v5_hn.py --models DCRNN-HAR --horizons 1 --seeds 1

This file loads the shared protocol from models/pga_trans_har.py;
no training starts on import.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.covariance import GraphicalLasso


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE.parent / "models" / "pga_trans_har.py"
if not PROTOCOL_PATH.is_file():
    raise FileNotFoundError(
        f"Required shared forecast protocol not found: {PROTOCOL_PATH}"
    )
spec = importlib.util.spec_from_file_location("pga_mse_retest_protocol", PROTOCOL_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load {PROTOCOL_PATH}")
protocol = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = protocol
spec.loader.exec_module(protocol)

TRADITIONAL = ("HAR", "VHAR", "HAR-KS")
DEEP = ("GNN-HAR", "DCRNN-HAR", "Pure-ST-Transformer")
ALL = (*TRADITIONAL, *DEEP)
FLOOR = float(protocol.QLIKE_VOL_FLOOR)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split_windows(values: np.ndarray, train_end: int, val_end: int,
                       horizon: int, seq_len: int):
    # Each target t has origin o=t-h. These first validation/test targets
    # therefore have origins exactly at the preceding split's last day.
    valid_start = train_end + horizon - 1
    test_start = val_end + horizon - 1
    train = protocol.make_windows(values, seq_len, train_end, seq_len, horizon)
    valid = protocol.make_windows(values, valid_start, val_end, seq_len, horizon)
    refit = protocol.make_windows(values, seq_len, val_end, seq_len, horizon)
    test = protocol.make_windows(values, test_start, len(values), seq_len, horizon)
    for label, windows, boundary in (
        ("validation", valid, train_end), ("test", test, val_end)
    ):
        if len(windows[4]) == 0:
            raise ValueError(f"No {label} samples at horizon {horizon}.")
        if int(windows[4][0]) - horizon != boundary - 1:
            raise AssertionError(f"{label} origin crosses the split boundary.")
    return train, valid, refit, test


def panel_features(windows):
    X, Xm, Y, Ym, indices = windows
    overlap, overlap_ok = protocol.har_features_numpy(X, Xm)
    x, m = X[..., 0], Xm[..., 0] > 0.5
    length, nodes = x.shape[1:]
    daily, daily_ok = protocol.numpy_last_valid(x, m)
    week, week_ok = protocol.numpy_masked_mean(x, m, length-5, length-1)
    month, month_ok = protocol.numpy_masked_mean(x, m, 0, length-5)
    nonoverlap = np.stack((daily, week, month), axis=-1)
    nonoverlap_ok = daily_ok & week_ok & month_ok
    kitchen = np.empty((len(X), nodes, nodes+2), dtype=np.float32)
    kitchen_ok = np.empty((len(X), nodes), dtype=bool)
    for node in range(nodes):
        kitchen[:, node] = np.concatenate(
            (overlap[:, node], np.delete(daily, node, axis=1)), axis=1
        )
        kitchen_ok[:, node] = overlap_ok[:, node] & np.delete(
            daily_ok, node, axis=1
        ).all(axis=1)
    return {
        "overlap": overlap, "overlap_ok": overlap_ok,
        "nonoverlap": nonoverlap.astype(np.float32),
        "nonoverlap_ok": nonoverlap_ok,
        "vhar": overlap.reshape(len(X), -1),
        "vhar_ok": overlap_ok.all(axis=1),
        "kitchen": kitchen, "kitchen_ok": kitchen_ok,
        "y": Y[..., 0], "ym": Ym[..., 0] > 0.5,
        "indices": indices,
    }


def ols_predict(design: np.ndarray, y: np.ndarray,
                fit: np.ndarray, predict: np.ndarray) -> np.ndarray:
    x_fit = np.column_stack((np.ones(int(fit.sum())), design[fit]))
    beta, *_ = np.linalg.lstsq(x_fit, y[fit], rcond=None)
    x_pred = np.column_stack((np.ones(int(predict.sum())), design[predict]))
    return x_pred @ beta


def predict_traditional(model: str, windows, val_end: int,
                        horizon: int) -> dict[str, np.ndarray]:
    feat = panel_features(windows)
    test = feat["indices"] >= val_end + horizon - 1
    fit = feat["indices"] < val_end
    n = feat["y"].shape[1]
    prediction = np.full((int(test.sum()), n), np.nan)
    eval_mask = np.zeros_like(prediction, dtype=bool)
    for node in range(n):
        if model == "HAR":
            design, good = feat["overlap"][:, node], feat["overlap_ok"][:, node]
        elif model == "VHAR":
            design, good = feat["vhar"], feat["vhar_ok"]
        else:
            design, good = feat["kitchen"][:, node], feat["kitchen_ok"][:, node]
        fit_node = fit & good & feat["ym"][:, node]
        if fit_node.sum() < design.shape[1] + 10:
            raise RuntimeError(f"Insufficient OLS observations: {model}, node {node}.")
        pred_node = test & good
        all_pred = np.full(len(test), np.nan)
        all_pred[pred_node] = ols_predict(
            design, feat["y"][:, node], fit_node, pred_node
        )
        prediction[:, node] = all_pred[test]
        eval_mask[:, node] = feat["ym"][test, node] & good[test]
    return {
        "prediction": np.where(np.isfinite(prediction),
                               np.maximum(prediction, FLOOR), np.nan),
        "target": feat["y"][test].astype(np.float64),
        "mask": eval_mask.astype(np.float32),
        "target_idx": feat["indices"][test],
    }


def glasso_graph(values: np.ndarray, alpha: float) -> np.ndarray:
    complete = protocol.causal_fill_window(values)
    z = (complete - complete.mean(axis=0)) / np.maximum(
        complete.std(axis=0, ddof=1), 1e-8
    )
    precision = GraphicalLasso(alpha=alpha, max_iter=2000).fit(z).precision_
    denominator = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
    graph = np.divide(np.abs(-precision), denominator,
                      out=np.zeros_like(precision), where=denominator > 0)
    np.fill_diagonal(graph, 1.0)
    graph /= np.maximum(graph.sum(axis=1, keepdims=True), 1e-12)
    return graph.astype(np.float32)


def sample_dy_graphs(values: np.ndarray, target_indices: np.ndarray,
                     horizon: int, seq_len: int, lags: int,
                     ridge: float, dy_horizon: int,
                     graph_cache: dict[int, np.ndarray]) -> np.ndarray:
    graphs = []
    for target_idx in target_indices:
        origin_end = int(target_idx) - horizon + 1
        if origin_end not in graph_cache:
            history = values[origin_end-seq_len:origin_end]
            if len(history) != seq_len:
                raise AssertionError("Graph history does not end at forecast origin.")
            graph = protocol.ridge_var_gfevd(
                history, lags=lags, ridge=ridge, horizon=dy_horizon
            )
            if not np.isfinite(graph).all():
                raise FloatingPointError(f"Nonfinite graph at origin {origin_end-1}.")
            graph_cache[origin_end] = graph
        graphs.append(graph_cache[origin_end])
    return np.stack(graphs, axis=0).astype(np.float32)


class ForecastSamples(Dataset):
    def __init__(self, windows, values: np.ndarray, horizon: int,
                 graphs: np.ndarray | None = None,
                 decoder_calendar: str = "observed"):
        X, Xm, Y, Ym, indices = windows
        self.x = torch.from_numpy(X.astype(np.float32))
        self.xm = torch.from_numpy(Xm.astype(np.float32))
        self.y = torch.from_numpy(Y.astype(np.float32))
        self.ym = torch.from_numpy(Ym.astype(np.float32))
        self.indices = torch.from_numpy(indices.astype(np.int64))
        self.graphs = None if graphs is None else torch.from_numpy(graphs)
        if self.graphs is not None and len(self.graphs) != len(self.x):
            raise ValueError("One dynamic graph is required per sample.")
        if horizon > 1:
            full_y, full_m = [], []
            for target_idx in indices:
                start = int(target_idx) - horizon + 1
                future = values[start:int(target_idx)+1]
                if len(future) != horizon:
                    raise AssertionError("Incomplete decoder target sequence.")
                full_m.append(np.isfinite(future).astype(np.float32))
                full_y.append(np.nan_to_num(future, nan=0.0).astype(np.float32))
            self.y_seq = torch.from_numpy(np.stack(full_y)[..., None])
            self.ym_seq = torch.from_numpy(np.stack(full_m)[..., None])
        else:
            self.y_seq, self.ym_seq = self.y[:, None], self.ym[:, None]
        if decoder_calendar == "observed":
            self.calendar = self.ym_seq
        elif decoder_calendar == "all_open":
            self.calendar = torch.ones_like(self.ym_seq)
        else:
            raise ValueError(f"Unknown decoder calendar: {decoder_calendar}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        graph = (self.graphs[i] if self.graphs is not None
                 else torch.empty(0, dtype=torch.float32))
        return (self.x[i], self.xm[i], self.y[i], self.ym[i],
                self.y_seq[i], self.ym_seq[i], self.calendar[i],
                self.indices[i], graph)


class GraphConv(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, adjacency):
        return F.relu(self.linear(torch.bmm(adjacency, x)))


class GNNHAR(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 16):
        super().__init__()
        self.num_nodes = num_nodes
        self.gcn1 = GraphConv(3, hidden_dim)
        self.gcn2 = GraphConv(hidden_dim, 1)
        self.har = nn.ModuleList([nn.Linear(3, 1) for _ in range(num_nodes)])

    def forward(self, x, xm, graph):
        overlap, nonoverlap = torch_har_features(x, xm)
        source_active = xm[:, -1, :, 0]
        adjacency = source_normalize(graph, source_active)
        gnn = self.gcn2(self.gcn1(nonoverlap, adjacency), adjacency)
        har = torch.cat(
            [self.har[n](overlap[:, n]) for n in range(self.num_nodes)], dim=1
        )[..., None]
        return F.softplus(har + gnn).clamp_min(FLOOR)


def torch_har_features(x, xm):
    # Overlapping HAR covariates: most recent valid, five-day and 22-day means.
    daily = protocol.torch_last_valid(x, xm)
    weekly = protocol.torch_masked_mean(x, xm, x.shape[1]-5, x.shape[1])
    monthly = protocol.torch_masked_mean(x, xm, 0, x.shape[1])
    overlap = torch.cat((daily, weekly, monthly), dim=-1)
    # Graph/HAR nonoverlapping covariates: 1, preceding 4, preceding 17.
    preweek = protocol.torch_masked_mean(x, xm, x.shape[1]-5, x.shape[1]-1)
    premonth = protocol.torch_masked_mean(x, xm, 0, x.shape[1]-5)
    nonoverlap = torch.cat((daily, preweek, premonth), dim=-1)
    return overlap, nonoverlap


def source_normalize(graph: torch.Tensor,
                     source_active: torch.Tensor) -> torch.Tensor:
    # graph[b,i,j]: node i receives from source j. Closed sources cannot send;
    # closed destinations may still receive. An empty row remains zero, so an
    # inactive node cannot leak through a diagonal fallback.
    if graph.ndim == 2:
        graph = graph[None].expand(source_active.size(0), -1, -1)
    masked = graph * source_active[:, None, :]
    degrees = masked.sum(dim=-1, keepdim=True)
    return masked / degrees.clamp_min(1e-12)


class DiffusionConv(nn.Module):
    """Forward K-term diffusion: orders 0 through K-1 (Chi et al., Eq. 18)."""
    def __init__(self, input_dim: int, output_dim: int, steps: int):
        super().__init__()
        if steps < 1:
            raise ValueError("diffusion steps must be positive")
        self.steps = steps
        self.linear = nn.Linear(input_dim * steps, output_dim)

    def forward(self, x, walk):
        terms = [x]
        state = x
        for _ in range(1,self.steps):
            state = torch.bmm(walk, state)
            terms.append(state)
        return self.linear(torch.cat(terms, dim=-1))


class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, steps: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = DiffusionConv(input_dim+hidden_dim, 2*hidden_dim, steps)
        self.candidate = DiffusionConv(input_dim+hidden_dim, hidden_dim, steps)

    def forward(self, x, hidden, walk):
        reset, update = torch.sigmoid(self.gates(
            torch.cat((x, hidden), dim=-1), walk
        )).chunk(2, dim=-1)
        candidate = torch.tanh(self.candidate(
            torch.cat((x, reset*hidden), dim=-1), walk
        ))
        return update*hidden + (1.0-update)*candidate


class DCRNNHAR(nn.Module):
    """Chi et al. encoder-decoder DCRNN-HAR with the aligned MSE objective."""
    def __init__(self, num_nodes: int, horizon: int, hidden_dim: int = 16,
                 diffusion_steps: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.encoder = DCGRUCell(1, hidden_dim, diffusion_steps)
        self.decoder = DCGRUCell(1, hidden_dim, diffusion_steps)
        self.projection = nn.Linear(hidden_dim, 1)
        self.har = nn.ModuleList([nn.Linear(3, 1) for _ in range(num_nodes)])

    def forward(self, x, xm, graph, future_calendar):
        batch, seq_len, nodes, _ = x.shape
        if nodes != self.num_nodes or future_calendar.shape[1] != self.horizon:
            raise ValueError("DCRNN input/decoder dimensions do not match horizon.")
        _, har_features = torch_har_features(x, xm)
        har = torch.cat(
            [self.har[n](har_features[:, n]) for n in range(nodes)], dim=1
        )[..., None]
        hidden = x.new_zeros(batch, nodes, self.hidden_dim)
        for day in range(seq_len):
            walk = source_normalize(graph, xm[:, day, :, 0])
            hidden = self.encoder(x[:, day], hidden, walk)
        decoder_input = x.new_zeros(batch, nodes, 1)  # y_hat_0 = 0
        predictions = []
        for step in range(self.horizon):
            # Only scheduled activity, not future observed volatility, is seen.
            active = future_calendar[:, step, :, 0]
            walk = source_normalize(graph, active)
            hidden = self.decoder(decoder_input, hidden, walk)
            forecast = F.softplus(self.projection(hidden) + har).clamp_min(FLOOR)
            predictions.append(forecast)
            decoder_input = forecast * active[..., None]
        return torch.stack(predictions, dim=1)


class PureSpatialAttention(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = hidden_dim ** -0.5

    def forward(self, x, xm):
        b, t, n, h = x.shape
        q = self.q(x).reshape(b*t, n, h)
        k = self.k(x).reshape(b*t, n, h)
        v = self.v(x).reshape(b*t, n, h)
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        active = xm[..., 0].reshape(b*t, n)
        weights = protocol.masked_softmax(scores,
                                          active[:, None, :].expand(-1, n, -1))
        return self.out(torch.bmm(self.dropout(weights), v).reshape(b,t,n,h))


class PureTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.temporal = protocol.MultiHeadTemporalAttention(
            hidden_dim, heads, dropout
        )
        self.spatial = PureSpatialAttention(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4*hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, xm):
        hidden = self.norm1(hidden + self.dropout(self.temporal(hidden, xm)))
        hidden = self.norm2(hidden + self.dropout(self.spatial(hidden, xm)))
        return self.norm3(hidden + self.ffn(hidden))


class PureSTTransformer(nn.Module):
    """Pure masked ST attention, with neither econometric graph nor HAR head."""
    def __init__(self, num_nodes: int, seq_len: int,
                 hidden_dim: int = 32, blocks: int = 2, heads: int = 4,
                 dropout: float = 0.10):
        super().__init__()
        self.input_projection = nn.Linear(2, hidden_dim)
        self.position = nn.Parameter(torch.randn(1,seq_len,1,hidden_dim)*0.02)
        self.node = nn.Parameter(torch.randn(1,1,num_nodes,hidden_dim)*0.02)
        self.blocks = nn.ModuleList(
            [PureTransformerBlock(hidden_dim, heads, dropout) for _ in range(blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim,max(hidden_dim//2,8)), nn.GELU(),
            nn.Linear(max(hidden_dim//2,8),1),
        )

    def forward(self, x, xm):
        hidden = self.input_projection(torch.cat((x,xm),dim=-1))
        hidden = hidden + self.position + self.node
        for block in self.blocks:
            hidden = block(hidden,xm)
        return F.softplus(self.head(hidden[:,-1])).clamp_min(FLOOR)


class NodeBalancedMSE(nn.Module):
    """Original training-standardized, equal-node MSE on volatility."""
    def __init__(self, node_scale: np.ndarray):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(node_scale, dtype=torch.float32))

    def forward(self, prediction, target, mask):
        active = mask
        error2 = (prediction-target).square()
        reduce_axes = tuple(range(active.ndim-2))
        counts = active.sum(dim=reduce_axes)
        node_loss = (
            (error2*active).sum(dim=reduce_axes)
            / counts.clamp_min(1.0)
        )
        valid = counts.squeeze(-1)>0
        if not torch.any(valid):
            raise RuntimeError("No observed targets in MSE batch.")
        standardized = node_loss.squeeze(-1)/self.scale.square().clamp_min(1e-8)
        return standardized[valid].mean()


def target_scales(values: np.ndarray, target_end: int, seq_len: int,
                  horizon: int) -> np.ndarray:
    windows = protocol.make_windows(values, seq_len, target_end, seq_len, horizon)
    y, active = windows[2][..., 0], windows[3][..., 0] > 0.5
    return np.asarray([
        max(float(np.std(y[active[:, n], n], ddof=1)), 1e-3)
        for n in range(y.shape[1])
    ], dtype=np.float32)


def make_loader(data: Dataset, batch_size: int, shuffle: bool,
                seed: int) -> DataLoader:
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                      drop_last=False, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def make_model(name: str, nodes: int, horizon: int, args, device):
    if name == "GNN-HAR":
        model = GNNHAR(nodes, args.gnn_hidden)
    elif name == "DCRNN-HAR":
        model = DCRNNHAR(nodes, horizon, args.dcrnn_hidden,
                         args.diffusion_steps)
    else:
        model = PureSTTransformer(nodes, args.seq_len, args.hidden_dim,
                                  args.blocks, args.heads, args.dropout)
    return model.to(device)


def forward_batch(model, model_name, batch, static_graph, device):
    x, xm, y, ym, yseq, ymseq, calendar, indices, sample_graph = batch
    x, xm = x.to(device), xm.to(device)
    y, ym = y.to(device), ym.to(device)
    if model_name == "DCRNN-HAR":
        graph = sample_graph.to(device)
        sequence = model(x, xm, graph, calendar.to(device))
        return sequence, yseq.to(device), ymseq.to(device), indices
    if model_name == "GNN-HAR":
        pred = model(x, xm, static_graph)
    else:
        pred = model(x, xm)
    return pred, y, ym, indices


def epoch_train(model, model_name, data_loader, graph, optimizer,
                objective, device, grad_clip: float) -> float:
    model.train()
    total, count = 0.0, 0
    for batch in data_loader:
        optimizer.zero_grad(set_to_none=True)
        p,y,m,_ = forward_batch(model,model_name,batch,graph,device)
        loss = objective(p,y,m)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training MSE.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),grad_clip)
        optimizer.step()
        total += float(loss.detach().cpu())
        count += 1
    return total/max(count,1)


@torch.no_grad()
def collect(model, model_name, data_loader, graph, device):
    model.eval()
    predictions, targets, masks, indices = [], [], [], []
    for batch in data_loader:
        p,y,m,idx = forward_batch(model,model_name,batch,graph,device)
        if model_name == "DCRNN-HAR":
            p,y,m = p[:,-1],y[:,-1],m[:,-1]
        predictions.append(p.cpu().numpy()[...,0])
        targets.append(y.cpu().numpy()[...,0])
        masks.append(m.cpu().numpy()[...,0])
        indices.append(idx.numpy())
    return {"prediction":np.concatenate(predictions),
            "target":np.concatenate(targets),
            "mask":np.concatenate(masks),
            "target_idx":np.concatenate(indices)}


def panel_mse_numpy(arrays, node_scale: np.ndarray) -> float:
    losses = []
    for node in range(arrays["target"].shape[1]):
        good = arrays["mask"][:,node]>0.5
        if good.any():
            losses.append(float(np.mean((
                arrays["target"][good,node] - arrays["prediction"][good,node]
            )**2)/max(float(node_scale[node])**2, 1e-8)))
    if not losses:
        raise RuntimeError("No active targets in validation set.")
    return float(np.mean(losses))


def fit_select(model, name, train_data, val_data, static_graph, args,
               device, seed, lr, node_scale):
    objective = NodeBalancedMSE(node_scale).to(device)
    optimizer = AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=args.max_epochs,eta_min=args.lr_min
    )
    train_loader = make_loader(train_data,args.batch_size,True,seed)
    val_loader = make_loader(val_data,args.batch_size,False,seed)
    best, best_epoch, stale, history = math.inf, 0, 0, []
    for epoch in range(1,args.max_epochs+1):
        train_loss = epoch_train(model,name,train_loader,static_graph,
                                 optimizer,objective,device,args.grad_clip)
        val_loss = panel_mse_numpy(collect(
            model,name,val_loader,static_graph,device
        ), node_scale)
        history.append({"epoch":epoch,"train_standardized_mse":train_loss,
                        "val_horizon_standardized_mse":val_loss,
                        "learning_rate":optimizer.param_groups[0]["lr"]})
        if val_loss<best-1e-9:
            best,best_epoch,stale = val_loss,epoch,0
        else:
            stale += 1
        scheduler.step()
        if stale>=args.patience:
            break
    if best_epoch<1:
        raise RuntimeError("Validation did not select an epoch.")
    return best_epoch,pd.DataFrame(history)


def refit_fixed_epochs(model, name, refit_data, static_graph, args,
                       device, seed, lr, epochs, node_scale):
    optimizer = AdamW(model.parameters(),lr=lr,weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=epochs,eta_min=args.lr_min
    )
    objective = NodeBalancedMSE(node_scale).to(device)
    train_loader = make_loader(refit_data,args.batch_size,True,seed)
    loss = float("nan")
    for _ in range(epochs):
        loss = epoch_train(model,name,train_loader,static_graph,
                           optimizer,objective,device,args.grad_clip)
        scheduler.step()
    return loss


def prediction_frame(name, seed, horizon, arrays, dates, names):
    rows = []
    for i,target_idx in enumerate(arrays["target_idx"]):
        target_idx = int(target_idx)
        origin_idx = target_idx-horizon
        for node,node_name in enumerate(names):
            active = bool(arrays["mask"][i,node]>0.5)
            rows.append({
                "model":name,"horizon":horizon,"seed":seed,
                "target_idx":target_idx,
                "target_date":pd.Timestamp(dates[target_idx]).date().isoformat(),
                "origin_idx":origin_idx,
                "origin_date":pd.Timestamp(dates[origin_idx]).date().isoformat(),
                "node":node,"node_name":node_name,
                "y_sqrt_rv_pct":float(arrays["target"][i,node]) if active else np.nan,
                "pred_sqrt_rv_pct":float(arrays["prediction"][i,node])
                                   if np.isfinite(arrays["prediction"][i,node]) else np.nan,
                "active":int(active),
            })
    return pd.DataFrame(rows)


def metric_table(name, seed, horizon, arrays, names):
    rows = []
    for node,node_name in enumerate(names):
        good = arrays["mask"][:,node]>0.5
        good &= np.isfinite(arrays["prediction"][:,node])
        y = arrays["target"][good,node]
        p = arrays["prediction"][good,node]
        if len(y)==0:
            raise RuntimeError(f"No evaluable targets: {name} {node_name}.")
        rows.append({
            "model":name,"horizon":horizon,"seed":seed,
            "node":node,"node_name":node_name,"n_active":len(y),
            "MSE":float(np.mean((p-y)**2)),
            "MAE":float(np.mean(np.abs(p-y))),
            "QLIKE":float(np.mean(protocol.qlike_from_volatility_numpy(y,p))),
        })
    return pd.DataFrame(rows)


def summarize(metrics):
    result = metrics.groupby(
        ["model","horizon","node","node_name"],as_index=False
    ).agg(n_seeds=("seed","nunique"),n_active=("n_active","min"),
          MSE_mean=("MSE","mean"),MSE_std=("MSE","std"),
          MAE_mean=("MAE","mean"),MAE_std=("MAE","std"),
          QLIKE_mean=("QLIKE","mean"),QLIKE_std=("QLIKE","std"))
    return result.fillna({"MSE_std":0.0,"MAE_std":0.0,"QLIKE_std":0.0})


@dataclass
class GraphSettings:
    glasso_alpha: float = 0.08
    dcrnn_var_lag: int = 1
    dcrnn_var_ridge: float = 0.0
    dcrnn_dy_horizon: int = 22


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=(
        HERE.parent/"data"/"df_union_sqrt.csv"))
    parser.add_argument("--input-format",choices=["vol_pct","sqrt_rv","rv"],
                        default="vol_pct")
    parser.add_argument("--outdir",type=Path,default=HERE.parent/"results"/"baselines")
    parser.add_argument("--models",nargs="+",choices=ALL,default=list(ALL))
    parser.add_argument("--horizons",nargs="+",type=int,default=[1,5,22])
    parser.add_argument("--seeds",nargs="+",type=int,default=[1,2,3,4,5])
    parser.add_argument("--train-ratio",type=float,default=0.60)
    parser.add_argument("--val-ratio",type=float,default=0.10)
    parser.add_argument("--seq-len",type=int,default=22)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--max-epochs",type=int,default=150)
    parser.add_argument("--patience",type=int,default=20)
    parser.add_argument("--gnn-lr",type=float,default=0.002)
    parser.add_argument("--dcrnn-lr",type=float,default=0.001)
    parser.add_argument("--transformer-lr",type=float,default=5e-4)
    parser.add_argument("--lr-min",type=float,default=1e-6)
    parser.add_argument("--weight-decay",type=float,default=1e-4)
    parser.add_argument("--grad-clip",type=float,default=1.0)
    parser.add_argument("--gnn-hidden",type=int,default=16)
    parser.add_argument("--dcrnn-hidden",type=int,default=16)
    parser.add_argument("--diffusion-steps",type=int,default=2)
    parser.add_argument("--hidden-dim",type=int,default=32)
    parser.add_argument("--blocks",type=int,default=2)
    parser.add_argument("--heads",type=int,default=4)
    parser.add_argument("--dropout",type=float,default=0.10)
    parser.add_argument("--glasso-alpha",type=float,default=0.08)
    parser.add_argument("--dcrnn-var-lag",type=int,default=1)
    parser.add_argument("--dcrnn-var-ridge",type=float,default=0.0)
    parser.add_argument("--dcrnn-dy-horizon",type=int,default=22)
    parser.add_argument("--dcrnn-decoder-calendar",
                        choices=["observed","all_open"],default="observed",
                        help=("observed reproduces the paper's future activity mask "
                              "and assumes these closures were known at origin; "
                              "all_open avoids that assumption as a robustness run"))
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seq_len!=22:
        raise ValueError("This benchmark requires the same 22-day lookback as PGA.")
    if not args.data.is_file():
        raise FileNotFoundError(f"Input panel does not exist: {args.data}")
    horizons = sorted(set(args.horizons))
    if any(h<1 for h in horizons) or not horizons:
        raise ValueError("Horizons must be positive.")
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds or any(seed<0 for seed in seeds):
        raise ValueError("Provide one or more nonnegative seeds.")
    panel = protocol.load_panel(args.data,args.input_format)
    values,dates,names = panel.values,panel.dates,panel.names
    train_end,val_end = protocol.fixed_split(
        len(values),args.train_ratio,args.val_ratio
    )
    if val_end+max(horizons)-1>=len(values):
        raise ValueError("Not enough test dates for the requested horizons.")
    device = protocol.choose_device(args.device)
    if "qlike" in args.outdir.name.lower():
        raise ValueError("Refusing to write MSE results into a QLIKE-named output directory.")
    args.outdir.mkdir(parents=True,exist_ok=True)
    all_metrics,all_predictions,all_histories,inventory = [],[],[],[]
    graph_settings = GraphSettings(args.glasso_alpha,args.dcrnn_var_lag,
                                   args.dcrnn_var_ridge,args.dcrnn_dy_horizon)
    graph_cache: dict[int,np.ndarray] = {}
    for horizon in horizons:
        print(f"Horizon h={horizon}",flush=True)
        train_w,val_w,refit_w,test_w = make_split_windows(
            values,train_end,val_end,horizon,args.seq_len
        )
        for name in TRADITIONAL:
            if name not in args.models:
                continue
            windows = protocol.make_windows(
                values,args.seq_len,len(values),args.seq_len,horizon
            )
            arrays = predict_traditional(name,windows,val_end,horizon)
            frame = prediction_frame(name,0,horizon,arrays,dates,names)
            metrics = metric_table(name,0,horizon,arrays,names)
            dest = args.outdir/f"h{horizon}"/name/"seed0"
            dest.mkdir(parents=True,exist_ok=True)
            frame.to_csv(dest/"predictions.csv",index=False)
            metrics.to_csv(dest/"metrics.csv",index=False)
            all_predictions.append(frame)
            all_metrics.append(metrics)
            inventory.append({"model":name,"horizon":horizon,"seed":0,
                              "fit":"OLS_volatility"})
        for name in DEEP:
            if name not in args.models:
                continue
            print(f"  {name}: preparing origin-aligned datasets",flush=True)
            graph_dev = graph_refit = None
            if name=="GNN-HAR":
                graph_dev = torch.from_numpy(glasso_graph(
                    values[:train_end],args.glasso_alpha
                )).to(device)
                graph_refit = torch.from_numpy(glasso_graph(
                    values[:val_end],args.glasso_alpha
                )).to(device)
            def make_dataset(windows):
                graph = None
                if name=="DCRNN-HAR":
                    graph = sample_dy_graphs(
                        values,windows[4],horizon,args.seq_len,
                        args.dcrnn_var_lag,args.dcrnn_var_ridge,
                        args.dcrnn_dy_horizon,graph_cache
                    )
                return ForecastSamples(windows,values,horizon,graph,
                                       args.dcrnn_decoder_calendar)
            train_data = make_dataset(train_w)
            val_data = make_dataset(val_w)
            refit_data = make_dataset(refit_w)
            test_data = make_dataset(test_w)
            scale_dev = target_scales(values, train_end, args.seq_len, horizon)
            scale_refit = target_scales(values, val_end, args.seq_len, horizon)
            lr = (args.gnn_lr if name=="GNN-HAR" else
                  args.dcrnn_lr if name=="DCRNN-HAR" else
                  args.transformer_lr)
            for seed in seeds:
                print(f"    seed={seed}: select then refit",flush=True)
                seed_everything(seed)
                model = make_model(name,len(names),horizon,args,device)
                selected_epoch,history = fit_select(
                    model,name,train_data,val_data,graph_dev,args,device,seed,lr,
                    scale_dev
                )
                del model
                seed_everything(seed)
                model = make_model(name,len(names),horizon,args,device)
                final_loss = refit_fixed_epochs(
                    model,name,refit_data,graph_refit,args,device,seed,lr,
                    selected_epoch,scale_refit
                )
                arrays = collect(model,name,make_loader(
                    test_data,args.batch_size,False,seed
                ),graph_refit,device)
                frame = prediction_frame(name,seed,horizon,arrays,dates,names)
                metrics = metric_table(name,seed,horizon,arrays,names)
                metrics["selected_epoch"] = selected_epoch
                metrics["final_train_standardized_mse"] = final_loss
                dest = args.outdir/f"h{horizon}"/name/f"seed{seed}"
                dest.mkdir(parents=True,exist_ok=True)
                frame.to_csv(dest/"predictions.csv",index=False)
                metrics.to_csv(dest/"metrics.csv",index=False)
                torch.save({"model_state_dict":model.state_dict(),
                            "model":name,"horizon":horizon,"seed":seed,
                            "selected_epoch":selected_epoch},
                           dest/"model.pt")
                history.insert(0,"model",name)
                history.insert(1,"horizon",horizon)
                history.insert(2,"seed",seed)
                history["selected_epoch"] = selected_epoch
                history.to_csv(dest/"validation_history.csv",index=False)
                all_histories.append(history)
                all_predictions.append(frame)
                all_metrics.append(metrics)
                inventory.append({"model":name,"horizon":horizon,
                                  "seed":seed,"selected_epoch":selected_epoch,
                                  "parameters":sum(p.numel() for p in model.parameters()
                                                   if p.requires_grad),
                                  "fit":"node_balanced_MSE"})
    metrics = pd.concat(all_metrics,ignore_index=True)
    predictions = pd.concat(all_predictions,ignore_index=True)
    metrics.to_csv(args.outdir/"all_seed_metrics_ALL_HORIZONS.csv",index=False)
    predictions.to_csv(args.outdir/"all_seed_predictions_ALL_HORIZONS.csv",index=False)
    summarize(metrics).to_csv(args.outdir/"summary_mean_std_ALL_HORIZONS.csv",
                              index=False)
    pd.DataFrame(inventory).to_csv(args.outdir/"model_inventory.csv",index=False)
    if all_histories:
        pd.concat(all_histories,ignore_index=True).to_csv(
            args.outdir/"validation_history_ALL_HORIZONS.csv",index=False
        )
    config = {key:(str(value) if isinstance(value,Path) else value)
              for key,value in vars(args).items()}
    config.update({
        "data_resolved":str(args.data.resolve()),
        "data_sha256":hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "rows":len(values),"nodes":len(names),"names":list(names),
        "zero_codes_restored_to_nan":panel.zero_count,
        "train_end_exclusive":train_end,"validation_end_exclusive":val_end,
        "test_first_target_date_by_horizon":{
            str(h):str(dates[val_end+h-1].date()) for h in horizons},
        "split_protocol":"60_train_10_validation_30_test",
        "refit_on_first_70_percent":True,
        "neural_primary_loss":"equal-node training-standardized MSE on volatility",
        "neural_validation":"equal-node training-standardized MSE at the requested horizon",
        "traditional_fit":"OLS on volatility; QLIKE is reported only",
        "dcrnn_graph":"one lookback-only GFEVD per forecast origin",
        "dcrnn_graph_settings":asdict(graph_settings),
        "dcrnn_future_calendar_assumption":(
            "observed mask treated as a pre-announced trading schedule"
            if args.dcrnn_decoder_calendar=="observed" else
            "no future schedule supplied; all decoder days treated as open"
        ),
        "reported_metrics":["MSE","MAE","QLIKE"],
        "qlike_floor_volatility_pct":FLOOR,
        "device_used":str(device),
    })
    with (args.outdir/"run_config.json").open("w",encoding="utf-8") as stream:
        json.dump(config,stream,indent=2,ensure_ascii=False)
    print(f"Saved baseline results to {args.outdir.resolve()}",flush=True)


if __name__=="__main__":
    main()
