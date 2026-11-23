"""Stage 1 (v2): simulate dates, condition each shot once per preprocessing variant,
form full / half stacks with and without onset alignment, extract window features.
usage: python gen_cache.py main 0 1 2 ...   |   python gen_cache.py sweep 0 1 ..."""
import sys, os, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import dassim as D
import pipeline as P

OUT = '/home/claude/exp/cache'
os.makedirs(OUT, exist_ok=True)
TAU = 0.2
NUIS = dict(t_shift=-0.06, coupling_drift=0.15, noise_scale=1.3)
DISTAL_X = float(D.XS[1790])
SCEN = {
    'S1_repeat': dict(),
    'S2_nuisance': dict(**NUIS),
    'S3_plume': dict(perturb={'strength': 1.0}),
    'S4_plume_nuis': dict(perturb={'strength': 1.0}, **NUIS),
    'S5_distal_nuis': dict(perturb={'strength': 1.0, 'plume_x': DISTAL_X}, **NUIS),
}
VARIANTS = {'full': (True, True), 'nocmnr': (False, True), 'nodwt': (True, False), 'none': (False, False)}


def cond_shots(R, c, d):
    out = []
    for x in R:
        y = P.cmnr(x) if c else x
        y = P.dwt_denoise(y) if d else y
        out.append(y)
    return out


def labels(gt, shift=0):
    if gt is None:
        z = np.zeros(P.NTW * P.NSW)
        return z.astype(bool), z
    d = np.roll(gt['diff'], -shift, 0) if shift else gt['diff']
    b = np.roll(gt['bg'], -shift, 0) if shift else gt['bg']
    return P.window_labels(d, b + gt['noise_var'], TAU)


def aligned_stack(shots, ref_env, max_lag=600):
    sh = [P.estimate_shift(P.envelope_trace(y), ref_env, max_lag) for y in shots]
    return np.mean([np.roll(y, -l, 0) for y, l in zip(shots, sh)], 0), sh


def run_main(seed, scen=SCEN, variants=VARIANTS, tag='main'):
    fn = f'{OUT}/{tag}_{seed}.npz'
    if os.path.exists(fn):
        return
    t0 = time.time()
    acq = D.Acquisition(seed)
    rng = np.random.default_rng(1000 + seed)
    res = dict(src_ch=int(np.argmin(np.abs(D.XS - acq.src_x))))
    Rb, _ = D.date_records(acq, 4, rng)
    refs = {}
    for v, (c, d) in variants.items():
        sh = cond_shots(Rb, c, d)
        gb = np.mean(sh, 0)
        refs[v] = P.envelope_trace(gb)
        res[f'Fb_{v}'] = P.window_features(gb)
        res[f'FA_{v}'] = P.window_features(np.mean(sh[:2], 0))
        res[f'FB_{v}'] = P.window_features(np.mean(sh[2:], 0))
        if seed == 0 and v == 'full' and tag == 'main':
            np.savez_compressed(f'{OUT}/gathers_base.npz', raw=Rb[0][:, ::2], cmnr=P.cmnr(Rb[0])[:, ::2],
                                dwt=sh[0][:, ::2], stack=gb[:, ::2])
        del sh
    del Rb
    for name, kw in scen.items():
        R, gt = D.date_records(acq, 5, rng, **kw)
        y, ratio = labels(gt)
        res[f'{name}_y'] = y
        res[f'{name}_ratio'] = ratio
        for v, (c, d) in variants.items():
            if v != 'full' and name not in ('S3_plume', 'S4_plume_nuis'):
                continue
            sh = cond_shots(R, c, d)
            g = np.mean(sh, 0)
            res[f'{name}_F_{v}'] = P.window_features(g)
            go, shifts = aligned_stack(sh, refs[v])
            res[f'{name}_Foa_{v}'] = P.window_features(go)
            if v == 'full':
                res[f'{name}_shifts'] = np.array(shifts)
                res[f'{name}_yoa'] = labels(gt, int(np.median(shifts)))[0]
                if seed == 0 and tag == 'main':
                    np.savez_compressed(f'{OUT}/gathers_{name}.npz', stack=g[:, ::2], stack_oa=go[:, ::2],
                                        gtdiff=(gt['diff'][:, ::2] if gt is not None else np.zeros(1)))
            del sh
        del R
    np.savez_compressed(fn, **res)
    print(f'{tag} seed {seed} done in {time.time() - t0:.0f}s', flush=True)


def run_sweep(seed):
    scen = {f's{s}': dict(perturb={'strength': s}, **NUIS) for s in [0.25, 0.5, 0.75, 1.0, 1.5]}
    run_main(seed, scen=scen, variants={'full': (True, True)}, tag='sweep')


if __name__ == '__main__':
    mode = sys.argv[1]
    for s in map(int, sys.argv[2:]):
        (run_main if mode == 'main' else run_sweep)(s)
