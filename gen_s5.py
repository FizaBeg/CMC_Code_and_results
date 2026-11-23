import sys, os, time, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/exp')
import numpy as np, dassim as D, pipeline as P, gen_cache as GC
X5 = 1060.0
for seed in map(int, sys.argv[1:]):
    t0 = time.time()
    fn = f'{GC.OUT}/main_{seed}.npz'
    res = dict(np.load(fn))
    acq = D.Acquisition(seed); rng = np.random.default_rng(1000 + seed)
    Rb, _ = D.date_records(acq, 4, rng)
    gb = np.mean(GC.cond_shots(Rb, True, True), 0); ref = P.envelope_trace(gb)
    assert np.allclose(P.window_features(gb), res['Fb_full'], rtol=1e-4, atol=1e-6), 'baseline replay mismatch'
    del Rb
    rng = np.random.default_rng(7000 + seed)
    R, gt = D.date_records(acq, 5, rng, perturb={'strength': 1.0, 'plume_x': X5, 'plume_z': 6.0}, **GC.NUIS)
    name = 'S5_distal_nuis'
    y, ratio = GC.labels(gt); res[f'{name}_y'] = y; res[f'{name}_ratio'] = ratio
    sh = GC.cond_shots(R, True, True)
    g = np.mean(sh, 0); res[f'{name}_F_full'] = P.window_features(g)
    go, shifts = GC.aligned_stack(sh, ref); res[f'{name}_Foa_full'] = P.window_features(go)
    res[f'{name}_shifts'] = np.array(shifts); res[f'{name}_yoa'] = GC.labels(gt, int(np.median(shifts)))[0]
    if seed == 0:
        np.savez_compressed(f'{GC.OUT}/gathers_{name}.npz', stack=g[:, ::2], stack_oa=go[:, ::2], gtdiff=gt['diff'][:, ::2])
    np.savez_compressed(fn, **res)
    print('s5', seed, y.mean(), round(time.time() - t0), flush=True)
