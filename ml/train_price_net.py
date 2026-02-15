
import os
import math
import time
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from scipy.stats import norm

# --- utility: analytic BS price (for reference/control) ---
def bs_call_price(S0, K, r, sigma, T):
    if T <= 0:
        return max(S0 - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

# --- MC label generator (antithetic + control variate using S_T) ---
def mc_label_price(S0, K, r, sigma, T, N=20000, rng=None):
    """Return MC estimate of price (scalar). Uses antithetic + control variate (control = S_T)."""
    if rng is None:
        rng = np.random.default_rng()
    half = (N + 1) // 2
    Zh = rng.standard_normal(size=half)
    Z = np.concatenate([Zh, -Zh])[:N]
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0)
    control = ST
    E_control = S0 * math.exp(r * T)
    cov = np.cov(payoff, control, ddof=1)
    cov_pc = cov[0, 1]
    var_c = cov[1, 1] if cov.shape == (2,2) else np.var(control, ddof=1)
    beta = cov_pc / var_c if var_c > 0 else 0.0
    adjusted = payoff - beta * (control - E_control)
    price = math.exp(-r * T) * adjusted.mean()
    return price

# --- Dataset: sample option parameters and compute MC labels ---
class PriceDataset(Dataset):
    def __init__(self, n_samples:int, mc_per_label:int=20000, rng_seed:int=1234):
        rng = np.random.default_rng(rng_seed)
        arr = []
        labels = []
        for i in range(n_samples):
            # sample a parameter vector (tune ranges to your use case)
            S0 = rng.uniform(50, 150)        # spot
            K  = rng.uniform(50, 150)        # strike
            r  = rng.uniform(0.0, 0.1)       # interest rate
            sigma = rng.uniform(0.05, 0.6)   # vol
            T = rng.uniform(0.01, 2.0)       # time to maturity (years)
            price = mc_label_price(S0, K, r, sigma, T, N=mc_per_label, rng=rng)
            # features: normalized set (use log-moneyness and scaled params)
            x = np.array([math.log(S0 / K), T, sigma, r, math.log(K)])  # example feature vector
            arr.append(x.astype(np.float32))
            labels.append(np.array([price], dtype=np.float32))
        self.X = np.vstack(arr)
        self.y = np.vstack(labels)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# --- Small Residual MLP ---
class PriceNet(nn.Module):
    def __init__(self, in_dim=5, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden]*n_layers
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.GELU())
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- training loop ---
def train(model, dl_train, dl_val, epochs=200, lr=3e-4, device='cpu'):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10, verbose=True)
    best_val = 1e9
    best_state = None
    es_patience = 25
    es_count = 0
    criterion = nn.MSELoss()
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        # val
        model.eval()
        with torch.no_grad():
            val_losses = []
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())
            val_loss = float(np.mean(val_losses))
        sched.step(val_loss)
        mean_train = float(np.mean(train_losses))
        print(f"Epoch {epoch} train={mean_train:.6g} val={val_loss:.6g}")
        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
            es_count = 0
        else:
            es_count += 1
        if es_count > es_patience:
            print("Early stopping")
            break
    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# --- evaluation helpers ---
def parity_plot_data(model, dataset, device='cpu'):
    model.eval()
    loader = DataLoader(dataset, batch_size=2048)
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            p = model(xb).cpu().numpy().reshape(-1)
            preds.append(p)
            trues.append(yb.numpy().reshape(-1))
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return preds, trues

# --- script entrypoint ---
def main():
    # generate dataset (small example)
    ds = PriceDataset(n_samples=2000, mc_per_label=20000, rng_seed=42)
    n_val = int(0.2 * len(ds))
    n_train = len(ds) - n_val
    ds_train, ds_val = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PriceNet(in_dim=5, hidden=128, n_layers=3, dropout=0.1)
    t0 = time.time()
    model = train(model, dl_train, dl_val, epochs=400, lr=1e-3, device=device)
    t1 = time.time()
    print("Training done, time:", t1-t0)
    preds, trues = parity_plot_data(model, ds_val, device=device)
    # quick metrics
    mse = np.mean((preds - trues)**2)
    rmse = math.sqrt(mse)
    print("Val RMSE:", rmse)
    # save model
    os.makedirs("ml/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "ml/checkpoints/price_net.pt")
    # example: show 10 parity pairs
    for p,t in zip(preds[:10], trues[:10]):
        print(f"pred {p:.6f} true {t:.6f}")
if __name__ == "__main__":
    main()
