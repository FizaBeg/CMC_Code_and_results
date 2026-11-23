"""Faithful re-implementation of the GR-MCAE pipeline (run_pipeline_master.py)
plus baselines, optional onset alignment, and evaluation utilities."""
import numpy as np
import pywt
import torch
import torch.nn as nn
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from dassim import DT, NT, NS

T_WIN, S_WIN, T_STEP, S_STEP = 250, 56, 125, 28
T_STARTS = np.arange(0, NT - T_WIN + 1, T_STEP)
S_STARTS = np.arange(0, NS - S_WIN + 1, S_STEP)
NTW, NSW = len(T_STARTS), len(S_STARTS)
torch.set_num_threads(2)


# ----------------------------------------------------------------- conditioning
def cmnr(x):
    return x - np.median(x, axis=1, keepdims=True)


def dwt_denoise(x, wavelet='db4', level=3):
    """Per-channel universal soft thresholding (vectorised over channels)."""
    coeffs = pywt.wavedec(x, wavelet, level=level, axis=0)
    thr = np.median(np.abs(coeffs[-1]), axis=0) / 0.6745 * np.sqrt(2 * np.log(x.shape[0]))
    out = [coeffs[0]] + [pywt.threshold(c, thr[None, :], mode='soft') for c in coeffs[1:]]
    return pywt.waverec(out, wavelet, axis=0)[:x.shape[0]].astype(np.float32)


def envelope_trace(x):
    return np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1))


def estimate_shift(env, ref_env, max_lag):
    """Integer lag maximising cross-correlation of channel-averaged envelopes."""
    a = (env - env.mean()) / (env.std() + 1e-12)
    b = (ref_env - ref_env.mean()) / (ref_env.std() + 1e-12)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = [np.dot(np.roll(a, -l), b) for l in lags]
    return int(lags[int(np.argmax(cc))])


def condition_date(records, use_cmnr=True, use_dwt=True, ref_env=None, max_lag=600):
    """CMNR -> DWT per shot, optional onset alignment, then arithmetic stack."""
    out = []
    shifts = []
    for x in records:
        y = cmnr(x) if use_cmnr else x
        y = dwt_denoise(y) if use_dwt else y
        if ref_env is not None:
            l = estimate_shift(envelope_trace(y), ref_env, max_lag)
            y = np.roll(y, -l, axis=0)
            shifts.append(l)
        out.append(y)
    return np.mean(out, axis=0), shifts


# ------------------------------------------------------------------- features
def window_features(g):
    """Five-moment descriptor for each 250x56 window (50% overlap).
    Vectorised: each window is the union of 2x2 non-overlapping 125x28 cells,
    so raw power sums are accumulated per cell and combined (exact)."""
    g = g[:T_STARTS[-1] + T_WIN, :S_STARTS[-1] + S_WIN].astype(np.float64)
    nt, ns = g.shape[0] // T_STEP, g.shape[1] // S_STEP
    c = g[:nt * T_STEP, :ns * S_STEP].reshape(nt, T_STEP, ns, S_STEP)
    sums = [c.sum((1, 3)), (c ** 2).sum((1, 3)), (c ** 3).sum((1, 3)), (c ** 4).sum((1, 3))]
    mx, mn = c.max((1, 3)), c.min((1, 3))
    comb = lambda a: a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]
    n = T_WIN * S_WIN
    m1, m2, m3, m4 = [comb(a) / n for a in sums]
    var = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    vmax = np.maximum(np.maximum(mx[:-1, :-1], mx[1:, :-1]), np.maximum(mx[:-1, 1:], mx[1:, 1:]))
    vmin = np.minimum(np.minimum(mn[:-1, :-1], mn[1:, :-1]), np.minimum(mn[:-1, 1:], mn[1:, 1:]))
    vs = np.maximum(var, 1e-30)
    F = np.stack([np.sqrt(m2), var, mu3 / vs ** 1.5, mu4 / vs ** 2 - 3.0, vmax - vmin], -1)
    return F.reshape(-1, 5)


def window_features_ref(g):
    F = np.zeros((NTW * NSW, 5), np.float64)
    k = 0
    for t in T_STARTS:
        blk_row = g[t:t + T_WIN].astype(np.float64)
        for s in S_STARTS:
            b = blk_row[:, s:s + S_WIN].ravel()
            F[k] = [np.sqrt(np.mean(b ** 2)), np.var(b), skew(b), kurtosis(b), b.max() - b.min()]
            k += 1
    return F


def window_coords():
    c = [[(t + t + T_WIN) / 2.0, (s + s + S_WIN) / 2.0] for t in T_STARTS for s in S_STARTS]
    return np.array(c)


def _win_mean(a):
    a = a[:T_STARTS[-1] + T_WIN, :S_STARTS[-1] + S_WIN].astype(np.float64)
    nt, ns = a.shape[0] // T_STEP, a.shape[1] // S_STEP
    c = a[:nt * T_STEP, :ns * S_STEP].reshape(nt, T_STEP, ns, S_STEP).sum((1, 3))
    return ((c[:-1, :-1] + c[1:, :-1] + c[:-1, 1:] + c[1:, 1:]) / (T_WIN * S_WIN)).ravel()


def window_labels(diff_energy, bg_energy, tau):
    """Ground truth: perturbation-to-background energy ratio above tau."""
    ratio = _win_mean(diff_energy) / (_win_mean(bg_energy) + 1e-12)
    return ratio > tau, ratio


# ------------------------------------------------------------------ autoencoder
class ProbabilisticFeatureAutoencoder(nn.Module):
    def __init__(self, input_dim=5, dropout_rate=0.15, h=3, z=2):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, h), nn.Tanh(), nn.Dropout(dropout_rate),
                                     nn.Linear(h, z), nn.Tanh())
        self.decoder = nn.Sequential(nn.Linear(z, h), nn.Tanh(), nn.Dropout(dropout_rate),
                                     nn.Linear(h, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_ae(X, seed, epochs=150, lr=0.01, p=0.15, val_mask=None):
    torch.manual_seed(seed)
    model = ProbabilisticFeatureAutoencoder(dropout_rate=p)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    Xtr = torch.tensor(X[~val_mask] if val_mask is not None else X, dtype=torch.float32)
    Xva = torch.tensor(X[val_mask], dtype=torch.float32) if val_mask is not None else None
    hist = []
    for e in range(epochs):
        model.train(); opt.zero_grad()
        loss = crit(model(Xtr), Xtr); loss.backward(); opt.step()
        if Xva is not None:
            model.eval()
            with torch.no_grad():
                vl = crit(model(Xva), Xva).item()
            hist.append((loss.item(), vl))
    return model, np.array(hist)


def mc_scores(model, X, M=30, seed=0):
    torch.manual_seed(seed + 12345)
    Xt = torch.tensor(X, dtype=torch.float32)
    model.train()  # dropout active
    with torch.no_grad():
        P = np.stack([np.mean((X - model(Xt).numpy()) ** 2, axis=1) for _ in range(M)])
    return P.mean(0), P.var(0)


def det_scores(model, X):
    model.eval()
    with torch.no_grad():
        return np.mean((X - model(torch.tensor(X, dtype=torch.float32)).numpy()) ** 2, axis=1)


# ------------------------------------------------------------------ graph
class GraphDiffuser:
    """(I + alpha L)^-1 s via eigendecomposition of the combinatorial Laplacian."""

    def __init__(self, coords, sigma_g=45.0, v_ref=150.0):
        c = coords.copy().astype(np.float64)
        c[:, 0] = c[:, 0] * DT * v_ref
        W = np.exp(-squareform(pdist(c, 'sqeuclidean')) / (2 * sigma_g ** 2))
        np.fill_diagonal(W, 0)
        L = np.diag(W.sum(1)) - W
        self.lam, self.V = np.linalg.eigh(L)
        self.lam = np.clip(self.lam, 0, None)

    def __call__(self, s, alpha):
        return self.V @ ((self.V.T @ s) / (1.0 + alpha * self.lam))


# ------------------------------------------------------------------ metrics
def jsd(p, q, bins=60):
    lo, hi = np.percentile(np.r_[p, q], [0.5, 99.5])
    hp, _ = np.histogram(np.clip(p, lo, hi), bins=bins, range=(lo, hi))
    hq, _ = np.histogram(np.clip(q, lo, hi), bins=bins, range=(lo, hi))
    hp = hp + 1e-12; hq = hq + 1e-12
    hp, hq = hp / hp.sum(), hq / hq.sum()
    m = 0.5 * (hp + hq)
    return 0.5 * (np.sum(hp * np.log2(hp / m)) + np.sum(hq * np.log2(hq / m)))


def morans_i(s):
    """Rook-contiguity Moran's I on the NTW x NSW window grid."""
    g = s.reshape(NTW, NSW)
    z = g - g.mean()
    num = (z[1:, :] * z[:-1, :]).sum() + (z[:, 1:] * z[:, :-1]).sum()
    w = (NTW - 1) * NSW + NTW * (NSW - 1)
    return (z.size / w) * num / ((z ** 2).sum() + 1e-12)
