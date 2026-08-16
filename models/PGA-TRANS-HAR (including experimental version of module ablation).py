#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from statsmodels.tsa.api import VAR
import copy


# In[2]:


# 1. Static graph construction: Estimating the directed static Diebold-Yilmaz volatility spillover matrix using VAR.
# =====================================================================
def compute_dy_adjacency(train_v_data, lags=2, horizon=10):
    model = VAR(train_v_data)
    results = model.fit(maxlags=lags)
    phi = results.ma_rep(horizon) 
    sigma = results.sigma_u        
    
    N = train_v_data.shape[1]
    theta = np.zeros((N, N))
    sigma_inv_diag = 1.0 / np.diag(sigma)
    
    for i in range(N):
        denom = 0.0
        for h in range(horizon):
            phi_h = phi[h] 
            term = phi_h @ sigma @ phi_h.T
            denom += term[i, i]
            
        for j in range(N):
            num = 0.0
            for h in range(horizon):
                phi_h = phi[h]
                term_num = phi_h @ sigma
                num += (term_num[i, j])**2
            theta[i, j] = sigma_inv_diag[j] * num / denom
            
    row_sums = theta.sum(axis=1, keepdims=True)
    norm_theta = theta / row_sums
    return norm_theta.T


# In[3]:


# 2. Spatial and Temporal Self-Attention Mechanism Layer (Based on Asymmetric Bias Initialization)
# =====================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, x, x_mask):
        batch_size, seq_len, N, hidden_dim = x.shape
        x_reshaped = x.transpose(1, 2).reshape(batch_size * N, seq_len, hidden_dim)
        mask_reshaped = x_mask.transpose(1, 2).reshape(batch_size * N, seq_len)
        
        q = self.W_q(x_reshaped) 
        k = self.W_k(x_reshaped)
        v = self.W_v(x_reshaped)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_dim) 
        temp_mask = (1.0 - mask_reshaped).unsqueeze(1) * -1e9 
        scores = scores + temp_mask
        
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v) 
        out = out.reshape(batch_size, N, seq_len, hidden_dim).transpose(1, 2)
        return out


class PriorGuidedSpatialAttention(nn.Module):

    def __init__(self, hidden_dim, num_nodes=8, mode='hybrid'):
        super(PriorGuidedSpatialAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.mode = mode
        
        self.S_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.S_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.S_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        if self.mode == 'hybrid':
            
            self.gate_alpha = nn.Parameter(torch.ones(num_nodes) * -1.38)
        
    def forward(self, x, x_mask, adj_prior):
        adj_prior = adj_prior.to(x.device)
        batch_size, seq_len, N, hidden_dim = x.shape
        
        x_reshaped = x.reshape(batch_size * seq_len, N, hidden_dim)
        mask_reshaped = x_mask.reshape(batch_size * seq_len, N)
        
        q = self.S_q(x_reshaped) 
        k = self.S_k(x_reshaped)
        v = self.S_v(x_reshaped)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_dim) 
        spat_mask = (1.0 - mask_reshaped).unsqueeze(1) * -1e9 
        scores = scores + spat_mask
        
        data_attn = torch.softmax(scores, dim=-1)
        prior_attn = adj_prior.unsqueeze(0).expand(batch_size * seq_len, -1, -1)
        
        if self.mode == 'no_prior':
            hybrid_attn = data_attn
        elif self.mode == 'static_prior':
            hybrid_attn = prior_attn
        else:
            
            gate = torch.sigmoid(self.gate_alpha).view(1, self.num_nodes, 1) 
            hybrid_attn = gate * data_attn + (1.0 - gate) * prior_attn
        
        out = torch.matmul(hybrid_attn, v)
        out = out.reshape(batch_size, seq_len, N, hidden_dim)
        return out


# In[4]:


# 3. The complete ST-Transformer-HAR model
# =====================================================================
class SpatioTemporalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_nodes=8, mode='hybrid'):
        super(SpatioTemporalTransformerBlock, self).__init__()
        self.temp_attn = TemporalAttention(hidden_dim)
        self.spat_attn = PriorGuidedSpatialAttention(hidden_dim, num_nodes=num_nodes, mode=mode)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, x_mask, adj_prior):
        temp_out = self.temp_attn(x, x_mask)
        x = self.norm1(x + temp_out)
        
        spat_out = self.spat_attn(x, x_mask, adj_prior)
        x = self.norm2(x + spat_out)
        
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class STTransformerHARModel(nn.Module):
    def __init__(self, num_nodes, in_dim=1, hidden_dim=16, seq_len=22, mode='hybrid'):
        super(STTransformerHARModel, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(in_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, 1, hidden_dim) * 0.02)
        
        self.st_transformer1 = SpatioTemporalTransformerBlock(hidden_dim, num_nodes=num_nodes, mode=mode)
        self.st_transformer2 = SpatioTemporalTransformerBlock(hidden_dim, num_nodes=num_nodes, mode=mode)
        
        self.output_projection = nn.Linear(hidden_dim, in_dim)
        
        self.har_regressors = nn.ModuleList([
            nn.Linear(3, 1) for _ in range(num_nodes)
        ])
        
    def forward(self, x, x_mask, adj_prior):
        x_emb = self.input_projection(x) + self.pos_embedding
        
        h_st = self.st_transformer1(x_emb, x_mask, adj_prior)
        h_st = self.st_transformer2(h_st, x_mask, adj_prior)
        
        h_last = h_st[:, -1, :, :]
        transformer_out = self.output_projection(h_last)
        
        x_d = x[:, -1, :, :]
        x_w = torch.mean(x[:, -5:-1, :, :], dim=1)
        x_m = torch.mean(x[:, -22:-5, :, :], dim=1)
        
        har_outs = []
        for j in range(self.num_nodes):
            node_features = torch.cat([x_d[:, j, :], x_w[:, j, :], x_m[:, j, :]], dim=-1)
            node_pred = self.har_regressors[j](node_features)
            har_outs.append(node_pred.unsqueeze(1))
            
        har_out = torch.cat(har_outs, dim=1)
        
        final_prediction = transformer_out + har_out
        final_prediction = torch.clamp(final_prediction, min=1e-3)
        return final_prediction


# In[5]:


# 4. Sliding Window and Loss Function
# =====================================================================
def prepare_st_transformer_dataset(raw_rv_data, seq_len=22, horizon=1):
    is_active = (~np.isnan(raw_rv_data)).astype(float)
    clean_rv_data = np.nan_to_num(raw_rv_data, nan=0.0)
    v = np.sqrt(clean_rv_data)
    T, N = v.shape
    
    X, X_mask = [], []
    Y, Y_mask = [], []
    
    for t in range(seq_len, T - horizon + 1):
        x_seq = v[t - seq_len : t].reshape(seq_len, N, 1)
        x_mask_seq = is_active[t - seq_len : t].reshape(seq_len, N, 1)
        
        y_target = v[t].reshape(N, 1)
        y_mask_target = is_active[t].reshape(N, 1)
        
        X.append(x_seq)
        X_mask.append(x_mask_seq)
        Y.append(y_target)
        Y_mask.append(y_mask_target)
        
    return np.array(X), np.array(X_mask), np.array(Y), np.array(Y_mask)


class STTransformerDataset(Dataset):
    def __init__(self, X, X_mask, Y, Y_mask):
        self.X = torch.FloatTensor(X)
        self.X_mask = torch.FloatTensor(X_mask)
        self.Y = torch.FloatTensor(Y)
        self.Y_mask = torch.FloatTensor(Y_mask)
        
    def __len__(self):
        return len(self.Y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.X_mask[idx], self.Y[idx], self.Y_mask[idx]


class MaskedQLIKELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskedQLIKELoss, self).__init__()
        self.eps = eps
        
    def forward(self, pred, target, mask):
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)
        
        loss = (target / pred) - torch.log(target / pred) - 1
        masked_loss = loss * mask
        
        denom = torch.sum(mask)
        if denom > 0:
            return torch.sum(masked_loss) / denom
        else:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)


# In[6]:


# 5. Training Pipeline: Equipped with a gradient separation engine (Curriculum Optimization) with a moderate learning rate
# =====================================================================
def run_st_transformer_pipeline(train_rv, val_rv, test_rv, norm_adj_tensor, num_nodes, mode='hybrid'):
    seq_len = 22
    X_train, X_mask_train, Y_train, Y_mask_train = prepare_st_transformer_dataset(train_rv, seq_len=seq_len)
    X_val, X_mask_val, Y_val, Y_mask_val = prepare_st_transformer_dataset(val_rv, seq_len=seq_len)
    X_test, X_mask_test, Y_test, Y_mask_test = prepare_st_transformer_dataset(test_rv, seq_len=seq_len)
    
    train_dataset = STTransformerDataset(X_train, X_mask_train, Y_train, Y_mask_train)
    val_dataset = STTransformerDataset(X_val, X_mask_val, Y_val, Y_mask_val)
    test_dataset = STTransformerDataset(X_test, X_mask_test, Y_test, Y_mask_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    
    model = STTransformerHARModel(num_nodes=num_nodes, in_dim=1, hidden_dim=16, seq_len=seq_len, mode=mode).to(device)
    norm_adj_tensor = norm_adj_tensor.to(device)
    
    criterion = MaskedQLIKELoss()
    
    if mode == 'hybrid':
        gate_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'gate_alpha' in name:
                gate_params.append(param)
            else:
                other_params.append(param)
        
        
        optimizer = optim.Adam([
            {'params': other_params, 'lr': 0.0005}, 
            {'params': gate_params, 'lr': 0.005}     
        ], weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, threshold=1e-4)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epochs = 150 
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, x_mask, y, y_mask in train_loader:
            x, x_mask, y, y_mask = x.to(device), x_mask.to(device), y.to(device), y_mask.to(device)
            
            optimizer.zero_grad()
            predictions = model(x, x_mask, norm_adj_tensor)
            
            loss = criterion(predictions, y, y_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_dataset)
        
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, x_mask, y, y_mask in val_loader:
                x, x_mask, y, y_mask = x.to(device), x_mask.to(device), y.to(device), y_mask.to(device)
                predictions = model(x, x_mask, norm_adj_tensor)
                loss = criterion(predictions, y, y_mask)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_dataset)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
    
    model.load_state_dict(best_model_wts)
    return model, test_loader, device


# In[7]:


# 6. One-click adaptive model evaluation module (outputs MSE, MAE, adaptively aligned to GPU devices)
# =====================================================================
def evaluate_model(model, test_loader, device, norm_adj_tensor):
    model.eval()
    
    norm_adj_tensor = norm_adj_tensor.to(device)
    
    all_preds, all_targets, all_masks = [], [], []
    with torch.no_grad():
        for x, x_mask, y, y_mask in test_loader:
            x, x_mask, y, y_mask = x.to(device), x_mask.to(device), y.to(device), y_mask.to(device)
            predictions = model(x, x_mask, norm_adj_tensor)
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(y_mask.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0).squeeze(-1)
    all_targets = np.concatenate(all_targets, axis=0).squeeze(-1)
    all_masks = np.concatenate(all_masks, axis=0).squeeze(-1)
    
    num_nodes = all_preds.shape[1]
    mse_results = []
    mae_results = []
    
    for i in range(num_nodes):
        mask_i = all_masks[:, i]
        active_indices = (mask_i == 1.0)
        
        active_pred = all_preds[:, i][active_indices]
        active_target = all_targets[:, i][active_indices]
        
        mse_results.append(np.mean((active_pred - active_target) ** 2))
        mae_results.append(np.mean(np.abs(active_pred - active_target)))
        
    return mse_results, mae_results


# In[8]:


# 7. One click operation of the main function for the complete ablation experiment (equipped with a brand new PGA v2 node autoregressive network)
# =====================================================================
def run_all_ablation_studies(data_path="df_union_sqrt.csv"):
    df = pd.read_csv(data_path, index_col=0)
    raw_rv_values = df.values
    num_nodes = raw_rv_values.shape[1]
    
    
    total_len = len(raw_rv_values)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.8)
    
    train_rv = raw_rv_values[:train_end]
    val_rv = raw_rv_values[train_end:val_end]
    test_rv = raw_rv_values[val_end:]
    
    
    
    train_df_filled = pd.DataFrame(train_rv).interpolate(method='linear', limit_direction='both').values
    train_v_filled = np.sqrt(train_df_filled)
    norm_adj_matrix = compute_dy_adjacency(train_v_filled, lags=2, horizon=10)
    norm_adj_tensor = torch.FloatTensor(norm_adj_matrix)
    
    
    
    model_hybrid, test_loader, device = run_st_transformer_pipeline(
        train_rv, val_rv, test_rv, norm_adj_tensor, num_nodes, mode='hybrid'
    )
    hybrid_mse, hybrid_mae = evaluate_model(model_hybrid, test_loader, device, norm_adj_tensor)
    
    
    
    model_noprior, _, _ = run_st_transformer_pipeline(
        train_rv, val_rv, test_rv, norm_adj_tensor, num_nodes, mode='no_prior'
    )
    noprior_mse, noprior_mae = evaluate_model(model_noprior, test_loader, device, norm_adj_tensor)
    
    
    
    model_static, _, _ = run_st_transformer_pipeline(
        train_rv, val_rv, test_rv, norm_adj_tensor, num_nodes, mode='static_prior'
    )
    static_mse, static_mae = evaluate_model(model_static, test_loader, device, norm_adj_tensor)
    
    # =====================================================================
    # 8. Print the ultimate ablation experiment comparison matrix report
    # =====================================================================
    indices_names = ['SPX', 'GDAXI', 'FCHI', 'FTSE', 'OMXSPI', 'N225', 'KS11', 'HSI']
    
    print("\n" +   "消融实验报告 (h=1) " )
    print(f"{'指数':<8} | {'无先验版(g=1) MSE/MAE':<24} | {'纯先验版(g=0) MSE/MAE':<24} | {'PGA改良版(自适应g) MSE/MAE':<24}")
    print("-" * 86)
    for i, name in enumerate(indices_names):
        noprior_str = f"{noprior_mse[i]:.4f} / {noprior_mae[i]:.4f}"
        static_str = f"{static_mse[i]:.4f} / {static_mae[i]:.4f}"
        hybrid_str = f"{hybrid_mse[i]:.4f} / {hybrid_mae[i]:.4f}"
        
        
        star = " ★" if (hybrid_mse[i] <= noprior_mse[i] and hybrid_mse[i] <= static_mse[i]) else ""
        
        print(f"{name:<8} | {noprior_str:<24} | {static_str:<24} | {hybrid_str:<24}{star}")
    

if __name__ == "__main__":
    run_all_ablation_studies(data_path="df_union_sqrt.csv")


# In[ ]:




