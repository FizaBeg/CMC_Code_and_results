"""
TL-GR-MCAE on the Svelvik field records (DASGAS HDF5 layout used by run_pipeline_master.py).

Usage:
  python run_field_tlgrmcae.py <RAW_DATA_DIR> <OUT_DIR> [--alpha 0.1] [--sigma 15] [--pre 0.3] [--post 0.7]

<RAW_DATA_DIR>/<date>/dphi/*.h5|*.hdf5 with datasets "data" [n_samples, n_channels] and
"header/dt", optional "header/dataScale". Dates: 20250724 (baseline), 20250726, 20250728.

Differences from the original field run:
  * Coarse onset detection per shot on the full record (trigger offsets of several seconds were
    observed between dates), then a fixed analysis window [onset - pre, onset + post] is cut;
    fine onset alignment (cross-correlation to the baseline envelope) is then applied.
  * Time-lapse residual features, robust re-centring + signed-log compression, half-stack null model,
    robust score standardisation, graph diffusion and a null-calibrated P95 threshold.
Outputs (per monitoring date): score maps, MC variance, decisions, and a JSON summary.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

DATES = ['20250724', '20250726', '20250728']


def rz(r):
    med = np.median(r, 0)
    return (r - med) / (1.4826 * np.median(np.abs(r - med), 0) + 1e-9)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def rs(s):
    med = np.median(s)
    return (s - med) / (1.4826 * np.median(np.abs(s - med)) + 1e-12)


def load_shot(path, pre, post, chunk=5000):
    with h5py.File(path, 'r') as f:
        d = f['data']
        dt = float(f['header/dt'][()])
        scale = float(f['header/dataScale'][()]) if 'header/dataScale' in f else 1.0
        n = d.shape[0]
        # coarse onset: channel-averaged RMS envelope of the CMNR'd record, read in chunks
        env = np.zeros(n)
        for a in range(0, n, chunk):
            x = d[a:a + chunk, :].astype(np.float32)
            if np.issubdtype(d.dtype, np.integer):
                x *= scale
            env[a:a + x.shape[0]] = np.sqrt(np.mean(P.cmnr(x).astype(np.float64) ** 2, 1))
        k = max(1, int(0.01 / dt))
        env_s = np.convolve(env, np.ones(k) / k, mode='same')
        onset = int(np.argmax(env_s))
        a = max(0, onset - int(pre / dt))
        b = a + int((pre + post) / dt)
        if b > n:
            b = n; a = b - int((pre + post) / dt)
        x = d[a:b, :].astype(np.float32)
        if np.issubdtype(d.dtype, np.integer):
            x *= scale
    return x, onset * dt, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('raw'); ap.add_argument('out')
    ap.add_argument('--alpha', type=float, default=0.1); ap.add_argument('--sigma', type=float, default=15.0)
    ap.add_argument('--pre', type=float, default=0.3); ap.add_argument('--post', type=float, default=0.7)
    ap.add_argument('--no-dwt', action='store_true'); ap.add_argument('--M', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    A = ap.parse_args()
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    shots, onsets = {}, {}
    for date in DATES:
        p = Path(A.raw) / date / 'dphi'
        files = sorted(f for f in os.listdir(p) if f.lower().endswith(('.h5', '.hdf5')))
        shots[date], onsets[date] = [], []
        for f in files:
            x, on, dt = load_shot(p / f, A.pre, A.post)
            y = P.cmnr(x)
            if not A.no_dwt:
                y = P.dwt_denoise(y)
            shots[date].append(y); onsets[date].append(on)
        print(date, len(files), 'shots, coarse onsets (s):', np.round(onsets[date], 3), flush=True)
    base = shots[DATES[0]]
    if len(base) < 2:
        raise SystemExit('baseline needs >= 2 shots for the half-stack null model')
    gb = np.mean(base, 0); ref = P.envelope_trace(gb)
    h = len(base) // 2
    FA = P.window_features(np.mean(base[:h], 0)); FB = P.window_features(np.mean(base[h:], 0))
    Fb = P.window_features(gb)
    mu_s = np.r_[FA, FB].mean(0); sd_s = np.r_[FA, FB].std(0) + 1e-12
    z = lambda F: (F - mu_s) / sd_s
    r0 = slog(rz(z(FA) - z(FB)))
    Xtr = np.r_[r0, -r0]
    # NOTE: window grid follows the cut length; recompute coordinates for this grid
    model, hist = P.train_ae(Xtr, A.seed)
    G = P.GraphDiffuser(P.window_coords(), sigma_g=A.sigma)
    sn, _ = P.mc_scores(model, r0, M=A.M, seed=A.seed)
    null = G(rs(sn), A.alpha); thr = float(np.percentile(null, 95))
    summary = dict(alpha=A.alpha, sigma_g=A.sigma, threshold=thr, onsets=onsets,
                   final_train_loss=float(hist[-1][0]) if len(hist) else None)
    for date in DATES[1:]:
        go = np.mean([np.roll(y, -P.estimate_shift(P.envelope_trace(y), ref, 600), 0) for y in shots[date]], 0)
        r = slog(rz(z(P.window_features(go)) - z(Fb)))
        mu, var = P.mc_scores(model, r, M=A.M, seed=A.seed)
        s = G(rs(mu), A.alpha); flag = s > thr
        np.savez_compressed(out / f'tlgrmcae_{date}.npz', score=s, mc_mean=mu, mc_var=var, flag=flag,
                            coords=P.window_coords())
        summary[date] = dict(flag_rate=float(flag.mean()), score_median=float(np.median(s)),
                             score_p95=float(np.percentile(s, 95)), var_median=float(np.median(var)))
        print(date, summary[date], flush=True)
    json.dump(summary, open(out / 'tlgrmcae_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
