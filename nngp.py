# PyTorch rewrite of the original JAX code (kept honest to the math/flow)

import os
import math
import numpy as np

import torch
from torch import Tensor
from tqdm import trange

# match JAX double precision
torch.set_default_dtype(torch.float64)
device = torch.device("cpu")

# ------------------------------
# Model / kernel definitions
# ------------------------------

depth = 10  # same global depth as original

def init_params(seed: int = 1023) -> Tensor:
    """Sample [a, b, noise] like the original nngp_params."""
    g = torch.Generator(device=device).manual_seed(seed)
    a = 0.75 + (1.25 - 0.75) * torch.rand((), generator=g, device=device)
    b = 0.10 + (0.50 - 0.10) * torch.rand((), generator=g, device=device)
    noise = 2.0 + (3.0 - 2.0) * torch.rand((), generator=g, device=device)
    params = torch.stack([a, b, noise]).requires_grad_(True)
    return params

def _unpack(params: Tensor):
    a, b, noise = params[0], params[1], params[2]
    return a, b, noise

@torch.no_grad()
def _abs_inplace(params: Tensor) -> Tensor:
    params.data = params.data.abs()
    return params

def nngp_cov(params: Tensor, X: Tensor, Y: Tensor) -> Tensor:
    """
    Compute the NNGP kernel matrix K(X, Y) using the same recursion
    as the original nngp_kernel (vectorized over all pairs).
    X: (n, d), Y: (m, d)
    returns: (n, m)
    """
    a, b, _ = _unpack(params)
    n, d = X.shape
    m, _ = Y.shape

    # base covariances
    Kxx = b + a * (X.pow(2).sum(dim=1) / d)        # (n,)
    Kyy = b + a * (Y.pow(2).sum(dim=1) / d)        # (m,)
    Kxy = b + a * (X @ Y.T) / d                    # (n, m)

    for _ in range(depth):
        S = torch.sqrt(Kxx[:, None] * Kyy[None, :])                  # (n, m)
        cor = torch.clamp(Kxy / S, -1.0 + 1e-16, 1.0 - 1e-16)        # (n, m)
        theta = torch.arccos(cor)
        trig = torch.sin(theta) + (math.pi - theta) * torch.cos(theta)
        Kxy = b + (a / (2 * math.pi)) * S * trig

        Kxx = b + (a / 2.0) * Kxx
        Kyy = b + (a / 2.0) * Kyy

    return Kxy

def nngp_var_diag(params: Tensor, X: Tensor) -> Tensor:
    """
    Diagonal of K(X, X) under the same recursion (variance vector).
    Equivalent to diag(nngp_cov(params, X, X)).
    """
    a, b, _ = _unpack(params)
    d = X.shape[1]
    K = b + a * (X.pow(2).sum(dim=1) / d)  # (n,)
    for _ in range(depth):
        K = b + (a / 2.0) * K
    return K  # (n,)

def nngp_predict(params: Tensor, Xtest: Tensor, Xtrain: Tensor, Ytrain: Tensor) -> Tensor:
    """
    Posterior mean at Xtest.
    """
    n = Ytrain.shape[0]
    _, _, noise = _unpack(params)

    K_DD = nngp_cov(params, Xtrain, Xtrain) + torch.eye(n, device=device) * noise
    K_DD_inv = torch.linalg.inv(K_DD)

    K_xD = nngp_cov(params, Xtest, Xtrain)
    proj = K_xD @ K_DD_inv
    return proj @ Ytrain  # (ntest, p)

def nngp_dist(params: Tensor, Xtest: Tensor, Xtrain: Tensor, Ytrain: Tensor):
    """
    Posterior mean and marginal variance (per test point).
    """
    n = Ytrain.shape[0]
    _, _, noise = _unpack(params)

    K_DD = nngp_cov(params, Xtrain, Xtrain) + torch.eye(n, device=device) * noise
    K_DD_inv = torch.linalg.inv(K_DD)

    K_xD = nngp_cov(params, Xtest, Xtrain)
    proj = K_xD @ K_DD_inv
    mu = proj @ Ytrain  # (ntest, p)

    # variance = k_xx - rowwise dot(proj, K_xD)
    k_xx = nngp_var_diag(params, Xtest)                      # (ntest,)
    quad = (proj * K_xD).sum(dim=1)                          # (ntest,)
    sig = k_xx - quad
    return mu, sig

# ------------------------------
# Loss & optimizer step (manual, like original)
# ------------------------------

def compute_lr(g: Tensor, scale: int = 2) -> Tensor:
    # replicate: 10^(-floor(log10(abs(g))) - scale)
    gabs = g.abs().clamp_min(1e-30)
    return (10.0 ** (-torch.floor(torch.log10(gabs)) - scale))

def nll_loss(params: Tensor, Xtrain: Tensor, Ytrain: Tensor) -> Tensor:
    """
    Negative log likelihood per output dimension (same as original).
    """
    n, p = Ytrain.shape
    a, b, noise = _unpack(params)

    K = nngp_cov(params, Xtrain, Xtrain) + torch.eye(n, device=device) * noise
    K_inv = torch.linalg.inv(K)

    # sum over columns of Y: sum_i y_i^T K^{-1} y_i
    mse = (Ytrain * (K_inv @ Ytrain)).sum()

    # p * log|K|
    sign, logdet = torch.linalg.slogdet(K)
    pen = p * logdet
    nor = (p / 2.0) * math.log(2.0 * math.pi)

    return (0.5 * mse + 0.5 * pen + nor) / p

def gradient_step(params: Tensor, Xtrain: Tensor, Ytrain: Tensor) -> Tensor:
    if params.grad is not None:
        params.grad.zero_()
    loss = nll_loss(params, Xtrain, Ytrain)
    loss.backward()

    with torch.no_grad():
        lrs = compute_lr(params.grad, scale=2)
        params -= lrs * params.grad
        _abs_inplace(params)  # enforce positivity like tree_map(abs, params)

    return params

# ------------------------------
# Data loading (paths & shapes preserved)
# ------------------------------

nmod, n_train, n_test, nlat, nlon = 5, 600, 408, 36, 72  # e.g., 50y hist + 34y scen (monthly)
rng = np.random.default_rng(0)

xhist = [rng.normal(0, 1, (n_train, nlat, nlon)).astype(np.float64) for _ in range(nmod)]
xrcp  = [rng.normal(0, 1, (n_test,  nlat, nlon)).astype(np.float64) for _ in range(nmod)]

# os.makedirs("../submit/data/saved", exist_ok=True)
# pickle.dump(xhist, open("../submit/data/saved/xhist_tas.pkl", "wb"))
# pickle.dump(xrcp,  open("../submit/data/saved/xrcp_tas.pkl",  "wb"))

# xhist = pickle.load(open('../submit/data/saved/xhist_tas.pkl', 'rb'))  # list/array: [model][time, lat, lon]
# xrcp  = pickle.load(open('../submit/data/saved/xrcp_tas.pkl',  'rb'))  # list/array: [model][time, lat, lon]

nval  = 72
nmod  = len(xhist)
ntrain = xhist[0].shape[0]
ntest  = xrcp[0].shape[0]

# ------------------------------
# Experiments (training loop & predictions)
# ------------------------------

sgpr_list = []
sgpr_var_list = []

for m1 in trange(nmod):
    _, nlat, nlon = xhist[m1].shape

    # ---------- build training set ----------
    xtrain_blocks = []
    for m2 in range(nmod):
        if m1 != m2:
            x1 = xhist[m2].reshape(ntrain, -1)              # (ntrain, F)
            x2 = xrcp[m2][0:nval].reshape(nval, -1)         # (nval, F)
            xtrain_blocks.append(np.vstack([x1, x2]))       # (ntrain+nval, F)

    # design for linear trend (means per block)
    xmean_train = np.array([blk.mean(axis=1) for blk in xtrain_blocks])  # (#blocks, Ntot)
    Xtrain_np = np.hstack(xtrain_blocks)                                  # (Ntot, sumF)

    y1 = xhist[m1].reshape(ntrain, -1)
    y2 = xrcp[m1][0:nval].reshape(nval, -1)
    Ytrain_np = np.vstack([y1, y2])                                       # (Ntot, Fm1)

    # cast to torch
    Xtrain = torch.from_numpy(Xtrain_np).to(device)
    Ytrain = torch.from_numpy(Ytrain_np).to(device)

    ymean = Ytrain.mean(dim=1)                                            # (Ntot,)

    # multiple linear regression on block means
    Xtrend = torch.from_numpy(xmean_train).T.to(device)                   # (Ntot, #blocks)
    beta = torch.linalg.inv(Xtrend.T @ Xtrend) @ (Xtrend.T @ ymean)       # (#blocks,)
    Ytrain = Ytrain - (Xtrend @ beta)[:, None]                            # detrend

    # ---------- build test set ----------
    xtest_blocks = []
    for m2 in range(nmod):
        if m1 != m2:
            xtest_blocks.append(xrcp[m2][nval:ntest].reshape(ntest - nval, -1))

    xmean_test = np.array([blk.mean(axis=1) for blk in xtest_blocks])     # (#blocks, Ntest)
    Xtest_np = np.hstack(xtest_blocks)                                    # (Ntest, sumF)
    Xtest = torch.from_numpy(Xtest_np).to(device)
    Xtrend_test = torch.from_numpy(xmean_test).T.to(device)               # (Ntest, #blocks)

    # ---------- init & fit ----------
    params = init_params(seed=1023)  # same fixed seed per model as original
    iterations = 300
    for _ in trange(iterations, leave=False):
        params = gradient_step(params, Xtrain, Ytrain)

    # ---------- predict ----------
    sgpr_hat, sgpr_var = nngp_dist(params, Xtest, Xtrain, Ytrain)         # (Ntest, Fm1), (Ntest,)
    sgpr_hat = sgpr_hat + (Xtrend_test @ beta)[:, None]                   # add trend back
    sgpr_hat = sgpr_hat.reshape(-1, nlat, nlon)                           # (T, lat, lon)

    # predictive std: sqrt(var + noise)
    _, _, noise = _unpack(params)
    sgpr_predstd = torch.sqrt(sgpr_var + noise)                           # (Ntest,)

    sgpr_list.append(sgpr_hat.detach().cpu().numpy())
    sgpr_var_list.append(sgpr_predstd.detach().cpu().numpy())

# ------------------------------
# Save (same filenames/pattern)
# ------------------------------
os.makedirs('/submit/experiments/pred', exist_ok=True)
for t in trange(nmod):
    np.save(f'/submit/experiments/pred/sgpr_tas_{t}.npz',     sgpr_list[t])
    np.save(f'/submit/experiments/pred/sgpr_tas_{t}_var.npz', sgpr_var_list[t])
