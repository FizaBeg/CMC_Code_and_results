"""
Svelvik-geometry semi-synthetic DAS benchmark (SV-DASBench).

Generates active-source (hammer) DAS gathers on the Svelvik acquisition grid
(dt = 0.2 ms, dx = 1.0213 m, 2352 channels, 1.0 s analysis window = 5000
samples) with realistic nuisance factors and a controllable, localized
"plume-like" wavefield perturbation that provides window-level ground truth.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

DT = 2e-4
DX = 1.0213
NT = 5000
NS = 2352


def ricker(t, f):
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def fibre_offsets(ns=NS, dx=DX):
    """Map channel index to (along-trench x, depth z) coordinates.
    Trench segments are interrupted by three vertical borehole loops
    (down-going and up-going legs), mimicking a trench + well layout."""
    xs, zs = np.zeros(ns), np.zeros(ns)
    x, z = 0.0, 0.0
    # (start_channel, length, kind)
    loops = [(420, 160), (1050, 200), (1700, 180)]
    ch = 0
    mode = []
    for c in range(ns):
        mode.append('t')
    for s, L in loops:
        half = L // 2
        for c in range(s, s + half):
            mode[c] = 'd'
        for c in range(s + half, s + L):
            mode[c] = 'u'
    for c in range(ns):
        if mode[c] == 't':
            x += dx
        elif mode[c] == 'd':
            z += dx
        else:
            z -= dx
        xs[c], zs[c] = x, max(z, 0.0)
    return xs, zs


XS, ZS = fibre_offsets()


class Acquisition:
    """Static site properties shared by all dates (one random site per seed)."""

    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.seed = seed
        g = gaussian_filter1d(r.normal(0, 0.35, NS), 25)
        g = g / (g.std() + 1e-9) * 0.25
        self.coupling = np.exp(g)
        # poorly coupled segments
        for _ in range(4):
            c = r.integers(100, NS - 100)
            w = r.integers(10, 40)
            self.coupling[c:c + w] *= r.uniform(0.3, 0.6)
        self.dead = r.choice(NS, 12, replace=False)
        self.src_x = 970.0 + r.uniform(-10, 10)   # source along trench (m), ~60 m from the injection well loop
        self.t0 = 0.30                            # trigger time inside window (s)
        # scatterers for coda
        self.n_scat = 40
        self.scat_x = r.uniform(0, XS.max(), self.n_scat)
        self.scat_z = r.uniform(5, 60, self.n_scat)
        self.scat_a = r.normal(0, 0.08, self.n_scat)
        # plume (anomaly) zone: centred on the middle borehole loop
        self.plume_x = XS[1150] + r.uniform(-15, 15)
        self.plume_z = 22.0 + r.uniform(-4, 4)
        self.plume_r = 18.0


def _wavefield(acq, t0, src_dx, perturb, rng, amp_scale=1.0):
    """Noise-free wavefield (NT x NS) for one shot.
    perturb: None or dict(strength=..., kind=...)"""
    t = np.arange(NT) * DT
    sx = acq.src_x + src_dx
    dxs = XS - sx
    r = np.sqrt(dxs ** 2 + ZS ** 2) + 1.0
    U = np.zeros((NT, NS), dtype=np.float64)

    # spatial perturbation weight per channel (inside plume region)
    px = perturb.get('plume_x', acq.plume_x) if perturb else acq.plume_x
    pz = perturb.get('plume_z', acq.plume_z) if perturb else acq.plume_z
    dpl = np.sqrt((XS - px) ** 2 + (ZS - pz) ** 2)
    w_pl = np.exp(-0.5 * (dpl / acq.plume_r) ** 2) if perturb else np.zeros(NS)
    s = perturb['strength'] if perturb else 0.0

    modes = [  # velocity m/s, centre freq Hz, amplitude, attenuation 1/m
        (1500.0, 120.0, 0.6, 0.004),   # P / head wave
        (420.0, 70.0, 1.0, 0.004),     # S / guided
        (180.0, 35.0, 2.2, 0.004),     # surface (Rayleigh) wave
    ]
    for v, f, a, att in modes:
        # velocity reduction inside the plume (gas lowers vp, slows wave)
        tt = t0 + r / v
        if perturb:
            tt = tt + w_pl * s * 0.012          # up to 12 ms delay at s=1
        amp = a * np.exp(-att * r) / np.sqrt(r)
        if perturb:
            amp = amp * (1.0 - 0.45 * s * w_pl)  # attenuation in gas zone
        U += amp[None, :] * ricker(t[:, None] - tt[None, :], f)

    # layer reflections (two reflectors)
    for zr, v, f, a in [(35.0, 900.0, 90.0, 0.35), (70.0, 1300.0, 80.0, 0.22)]:
        rr = np.sqrt(dxs ** 2 + (2 * zr - ZS) ** 2) + 1.0
        tt = t0 + rr / v
        U += (a / np.sqrt(rr))[None, :] * ricker(t[:, None] - tt[None, :], f)

    # coda from point scatterers
    for k in range(acq.n_scat):
        d1 = np.hypot(acq.scat_x[k] - sx, acq.scat_z[k])
        d2 = np.hypot(XS - acq.scat_x[k], ZS - acq.scat_z[k]) + 1.0
        tt = t0 + (d1 + d2) / 300.0
        U += (acq.scat_a[k] / np.sqrt(d1 * d2 + 1.0) * 6)[None, :] * ricker(t[:, None] - tt[None, :], 55.0)

    # bubble/fluid-flow scattering inside plume: band-limited bursts after arrival
    if perturb:
        idx = np.where(w_pl > 0.05)[0]
        tarr = t0 + r[idx] / 180.0
        env = np.exp(-np.clip(t[:, None] - tarr[None, :], 0, None) / 0.18) * (t[:, None] > tarr[None, :])
        burst = rng.normal(0, 1, (NT, idx.size))
        # band-pass 150-600 Hz via difference of gaussians in time
        burst = gaussian_filter1d(burst, 1.2, axis=0) - gaussian_filter1d(burst, 5.0, axis=0)
        burst /= burst.std() + 1e-9
        U[:, idx] += 0.30 * s * (w_pl[idx] / np.sqrt(r[idx]))[None, :] * env * burst * 3.0

    return U * amp_scale


def shot_from_clean(U, acq, rng, noise_scale=1.0, coupling=None):
    """Add acquisition effects and noise to a noise-free wavefield."""
    jit = int(round(rng.normal(0, 0.0015) / DT))  # intra-date trigger jitter
    clean = np.roll(U, jit, axis=0) * coupling[None, :]
    sig_n = noise_scale * (1.0 + 0.3 * rng.random(NS))
    X = clean + rng.standard_normal((NT, NS), dtype=np.float32) * sig_n[None, :].astype(np.float32)
    cm = gaussian_filter1d(rng.normal(0, 1, NT), 3.0)
    cm = cm / cm.std() * 4.0 * noise_scale
    X += (cm[:, None] * (1.0 + 0.05 * rng.normal(0, 1, NS))[None, :]).astype(np.float32)
    hum_ch = rng.choice(NS, 300, replace=False)
    X[:, hum_ch] += (1.2 * noise_scale * np.sin(2 * np.pi * 50 * np.arange(NT) * DT + rng.uniform(0, 6.28)))[:, None].astype(np.float32)
    nsp = 400
    X[rng.integers(0, NT, nsp), rng.integers(0, NS, nsp)] += rng.normal(0, 25, nsp) * noise_scale
    X[:, acq.dead] = rng.normal(0, 0.05, (NT, acq.dead.size))
    return X.astype(np.float32), clean


def date_records(acq, n_shots, rng, perturb=None, t_shift=0.0, src_dx=0.0,
                 noise_scale=1.0, coupling_drift=0.0, signal_amp=40.0):
    """Simulate all shots of one acquisition date.
    Returns (records [n_shots, NT, NS], noise-free mean wavefield [NT, NS])."""
    U = (_wavefield(acq, acq.t0 + t_shift, src_dx, perturb, rng) * signal_amp).astype(np.float32)
    coup = acq.coupling.copy()
    if coupling_drift > 0:
        d = gaussian_filter1d(rng.normal(0, 1, NS), 40)
        d = d / (d.std() + 1e-9) * coupling_drift
        coup = coup * np.exp(d)
    recs, cl = [], np.zeros((NT, NS), np.float32)
    for _ in range(n_shots):
        x, c = shot_from_clean(U, acq, rng, noise_scale, coup.astype(np.float32))
        recs.append(x); cl += c / n_shots
    if perturb is not None:
        U0 = (_wavefield(acq, acq.t0 + t_shift, src_dx, None, rng) * signal_amp).astype(np.float32)
        gt = dict(diff=((U - U0) * coup[None, :]) ** 2, bg=(U0 * coup[None, :]) ** 2,
                  noise_var=(1.15 * noise_scale) ** 2 / n_shots)
    else:
        gt = None
    return np.stack(recs), gt
