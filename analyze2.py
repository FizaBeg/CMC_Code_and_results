"""Stage 2 (v2): scoring, metrics, ablations, sensitivity and statistics on SV-DASBench."""
import sys, os, json, time, warnings, glob
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import pipeline as P
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from scipy.stats import wilcoxon, spearmanr

C = '/home/claude/exp/cache'
R = '/home/claude/exp/results'
os.makedirs(R, exist_ok=True)
TEST = list(range(10))
VAL = [100, 101, 102]
SC = ['S1_repeat', 'S2_nuisance', 'S3_plume', 'S4_plume_nuis', 'S5_distal_nuis']
ANOM = ['S3_plume', 'S4_plume_nuis', 'S5_distal_nuis']
NULL = ['S1_repeat', 'S2_nuisance']
COORDS = P.window_coords()
VALMASK = np.tile(np.arange(P.NSW) >= int(0.8 * P.NSW), P.NTW)
DETS = ['Mahalanobis', 'PCA-Recon', 'IsolationForest', 'OC-SVM', 'LOF', 'Det-AE', 'MC-AE']
_G = {}


def G(sig):
    if sig not in _G:
        _G[sig] = P.GraphDiffuser(COORDS, sigma_g=sig)
    return _G[sig]


_cache = {}


def load(seed, tag='main'):
    k = (tag, seed)
    if k not in _cache:
        _cache[k] = dict(np.load(f'{C}/{tag}_{seed}.npz'))
    return _cache[k]


def rz(r):
    med = np.median(r, 0)
    mad = 1.4826 * np.median(np.abs(r - med), 0) + 1e-9
    return (r - med) / mad


def slog(x):
    """Signed-log compression of heavy-tailed residuals."""
    return np.sign(x) * np.log1p(np.abs(x))


def rs(s):
    """Date-wise robust standardisation of an anomaly-score map."""
    med = np.median(s)
    return (s - med) / (1.4826 * np.median(np.abs(s - med)) + 1e-12)


# ---------------------------------------------------------------- inputs
def make_inputs(d, mode, v='full', oa=True, recenter=True, feats=None, compress=True):
    """Returns (X_train, X_null, {scn: X_mon}, {scn: y}).
    mode 'TL': time-lapse residual features (proposed); 'ABS': absolute features (original)."""
    fi = slice(None) if feats is None else feats
    key = 'Foa' if oa else 'F'
    scns = [k[:-len(f'_{key}_{v}')] for k in d if k.endswith(f'_{key}_{v}') and not k.startswith('F')]
    ys = {s: (d[f'{s}_yoa'] if oa else d[f'{s}_y']) for s in scns}
    if mode == 'ABS':
        sc = StandardScaler().fit(d[f'Fb_{v}'][:, fi])
        Xb = sc.transform(d[f'Fb_{v}'][:, fi])
        return Xb, Xb, {s: sc.transform(d[f'{s}_{key}_{v}'][:, fi]) for s in scns}, ys
    s2 = StandardScaler().fit(np.r_[d[f'FA_{v}'], d[f'FB_{v}']][:, fi])
    f0 = rz if recenter else (lambda x: x)
    f = (lambda x: slog(f0(x))) if compress else f0
    rab = f(s2.transform(d[f'FA_{v}'][:, fi]) - s2.transform(d[f'FB_{v}'][:, fi]))
    zb = s2.transform(d[f'Fb_{v}'][:, fi])
    Xm = {s: f(s2.transform(d[f'{s}_{key}_{v}'][:, fi]) - zb) for s in scns}
    return np.r_[rab, -rab], rab, Xm, ys


def fit_detectors(Xtr, seed, which=DETS, p=0.15, M=30):
    out = {}
    if 'Mahalanobis' in which:
        out['Mahalanobis'] = EmpiricalCovariance().fit(Xtr).mahalanobis
    if 'PCA-Recon' in which:
        pca = PCA(2).fit(Xtr)
        out['PCA-Recon'] = lambda X: np.mean((X - pca.inverse_transform(pca.transform(X))) ** 2, 1)
    if 'IsolationForest' in which:
        ifo = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
        out['IsolationForest'] = lambda X: -ifo.score_samples(X)
    if 'OC-SVM' in which:
        oc = OneClassSVM(nu=0.05, gamma='scale').fit(Xtr)
        out['OC-SVM'] = lambda X: -oc.decision_function(X)
    if 'LOF' in which:
        lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(Xtr)
        out['LOF'] = lambda X: -lof.score_samples(X)
    model = hist = None
    if 'Det-AE' in which or 'MC-AE' in which:
        vm = np.r_[VALMASK, VALMASK] if len(Xtr) == 2 * len(VALMASK) else VALMASK
        model, hist = P.train_ae(Xtr, seed, p=p, val_mask=vm)
        out['Det-AE'] = lambda X: P.det_scores(model, X)
        out['MC-AE'] = lambda X: P.mc_scores(model, X, M=M, seed=seed)[0]
    return {k: v for k, v in out.items() if k in which}, model, hist


def decide(s_null, s_mon, alpha, sig, gr=True, norm=True):
    a = rs(s_null) if norm else s_null
    b = rs(s_mon) if norm else s_mon
    if gr:
        a, b = G(sig)(a, alpha), G(sig)(b, alpha)
    return a, b, np.percentile(a, 95)


def metr(y, s, thr, s_null):
    pred = s > thr
    r = dict(flag=float(pred.mean()), moran=float(P.morans_i(s)), jsd=float(P.jsd(s_null, s)))
    if y.any():
        r.update(auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s),
                 f1=f1_score(y, pred, zero_division=0), prec=precision_score(y, pred, zero_division=0),
                 rec=recall_score(y, pred, zero_division=0))
    return r


def run_config(seed, alpha, sig, mode='TL', v='full', oa=True, recenter=True, norm=True, feats=None,
               which=DETS, grs=(False, True), p=0.15, M=30, tag='main', compress=True):
    d = load(seed, tag)
    Xtr, Xn, Xm, ys = make_inputs(d, mode, v, oa, recenter, feats, compress)
    dets, model, hist = fit_detectors(Xtr, seed, which, p, M)
    out = {}
    for k, f in dets.items():
        sn = f(Xn)
        for scn, X in Xm.items():
            sm = f(X)
            for gr in grs:
                a, b, thr = decide(sn, sm, alpha, sig, gr, norm)
                out[(scn, ('GR-' if gr else '') + k)] = metr(ys[scn], b, thr, a)
    return out, model, hist


def agg(lst):
    keys = set().union(*[r.keys() for r in lst])
    out = {}
    for k in keys:
        rr = [r[k] for r in lst if k in r]
        ms = set().union(*[x.keys() for x in rr])
        out[k] = {m: (float(np.nanmean([x.get(m, np.nan) for x in rr])), float(np.nanstd([x.get(m, np.nan) for x in rr])))
                  for m in ms}
    return out


def tostr(d):
    return {f'{a}|{b}': v for (a, b), v in d.items()}


def sel():
    s = json.load(open(f'{R}/selected.json'))
    return s['alpha'], s['sigma_g']


if __name__ == '__main__':
    step = sys.argv[1]
    t0 = time.time()

    if step == 'select':
        alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        sigs = [5.0, 7.5, 10.0, 15.0, 22.5, 30.0, 45.0, 60.0, 90.0]
        grid = {}
        for seed in VAL + TEST:
            d = load(seed)
            Xtr, Xn, Xm, ys = make_inputs(d, 'TL')
            dets, _, _ = fit_detectors(Xtr, seed, ['MC-AE'])
            sn = dets['MC-AE'](Xn)
            sms = {s: dets['MC-AE'](Xm[s]) for s in SC}
            for sg in sigs:
                for a in alphas:
                    for s in SC:
                        an, b, thr = decide(sn, sms[s], a, sg)
                        r = metr(ys[s], b, thr, an)
                        grid[f'{seed}|{s}|{a}|{sg}'] = r
            print('select', seed, round(time.time() - t0), flush=True)
        json.dump(grid, open(f'{R}/grid.json', 'w'))
    if step in ('select', 'reselect'):
        grid = json.load(open(f'{R}/grid.json'))
        alphas = sorted({float(k.split('|')[2]) for k in grid}); sigs = sorted({float(k.split('|')[3]) for k in grid})
        score = {}
        for a in alphas:
            for sg in sigs:
                f1 = np.mean([grid[f'{s}|{c}|{a}|{sg}']['f1'] for s in VAL for c in ANOM])
                far = np.mean([grid[f'{s}|{c}|{a}|{sg}']['flag'] for s in VAL for c in NULL])
                score[(a, sg)] = (f1, far)
        # rule: maximise validation F1 subject to a null false-alarm rate <= 1.5 x nominal (0.075)
        adm = {k: v for k, v in score.items() if v[1] <= 0.075}
        best = max(adm, key=lambda k: adm[k][0])
        json.dump({'alpha': best[0], 'sigma_g': best[1], 'val_f1': score[best][0], 'val_far': score[best][1],
                   'table': {f'{a}|{sg}': v for (a, sg), v in score.items()}}, open(f'{R}/selected.json', 'w'), indent=1)
        print('best', best, score[best])

    if step == 'main':
        A, S = sel()
        allres, hists, unc, times = [], [], [], []
        for seed in TEST:
            d = load(seed)
            r, model, hist = run_config(seed, A, S, 'TL')
            hists.append(hist)
            r2, _, _ = run_config(seed, A, S, 'ABS', oa=False)       # original framework inputs
            r.update({(k[0], 'ABS:' + k[1]): v for k, v in r2.items()})
            # classical 4D RMS difference
            nul = np.abs(np.log(d['FA_full'][:, 0] + 1e-9) - np.log(d['FB_full'][:, 0] + 1e-9))
            for scn in SC:
                ys = d[f'{scn}_yoa']
                sm = np.abs(np.log(d[f'{scn}_Foa_full'][:, 0] + 1e-9) - np.log(d['Fb_full'][:, 0] + 1e-9))
                for gr in (False, True):
                    a, b, thr = decide(nul, sm, A, S, gr)
                    r[(scn, ('GR-' if gr else '') + '4D-dRMS')] = metr(ys, b, thr, a)
                # source-proximity diagnostic (oracle, not a detector)
                if ys.any():
                    prox = -np.abs(COORDS[:, 1] - d['src_ch'])
                    r[(scn, 'SourceProximity')] = dict(auroc=roc_auc_score(d[f'{scn}_y'], prox),
                                                       auprc=average_precision_score(d[f'{scn}_y'], prox))
            allres.append(r)
            # uncertainty analysis (proposed)
            Xtr, Xn, Xm, ys = make_inputs(d, 'TL')
            mn, vn = P.mc_scores(model, Xn, seed=seed)
            for scn in SC:
                t1 = time.perf_counter()
                mu, var = P.mc_scores(model, Xm[scn], seed=seed)
                a, b, thr = decide(mn, mu, A, S)
                times.append(time.perf_counter() - t1)
                y = ys[scn]
                pred = b > thr
                e = dict(seed=seed, scn=scn, rho=float(spearmanr(mu, var)[0]),
                         var_flag=float(np.median(var[pred])) if pred.any() else np.nan,
                         var_unflag=float(np.median(var[~pred])))
                if y.any():
                    e.update(auroc_var=roc_auc_score(y, var), auroc_mu=roc_auc_score(y, mu),
                             var_tp=float(np.median(var[pred & y])) if (pred & y).any() else np.nan,
                             var_fp=float(np.median(var[pred & ~y])) if (pred & ~y).any() else np.nan)
                unc.append(e)
            print('main', seed, round(time.time() - t0), flush=True)
        json.dump(tostr(agg(allres)), open(f'{R}/main_agg.json', 'w'), indent=1)
        json.dump([tostr(r) for r in allres], open(f'{R}/main_raw.json', 'w'))
        np.save(f'{R}/hists.npy', np.array(hists))
        json.dump(unc, open(f'{R}/uncertainty.json', 'w'), indent=1)
        json.dump(dict(mean_s=float(np.mean(times)), std_s=float(np.std(times))), open(f'{R}/infer_time.json', 'w'))
        prop = 'GR-MC-AE'
        tests = {}
        others = sorted({k for r in allres for (s_, k) in r if s_ == 'S4_plume_nuis' and k not in (prop, 'SourceProximity')})
        for other in others:
            for met in ('auroc', 'auprc', 'f1'):
                a = np.array([r[(c, prop)][met] for r in allres for c in ANOM])
                b = np.array([r[(c, other)][met] for r in allres for c in ANOM])
                try:
                    pv = float(wilcoxon(a, b).pvalue)
                except ValueError:
                    pv = 1.0
                tests[f'{other}|{met}'] = (float(np.mean(a - b)), pv)
        json.dump(tests, open(f'{R}/wilcoxon.json', 'w'), indent=1)

    if step == 'ablate':
        A, S = sel()
        out = {}
        cfgs = {
            'Proposed (full)': dict(),
            'w/o graph regularization': dict(grs=(False,)),
            'w/o MC Dropout (deterministic AE)': dict(which=['Det-AE']),
            'w/o onset alignment': dict(oa=False),
            'w/o robust re-centring': dict(recenter=False),
            'w/o signed-log compression': dict(compress=False),
            'w/o score standardisation': dict(norm=False),
            'w/o time-lapse referencing (original)': dict(mode='ABS', oa=False),
            'w/o CMNR': dict(v='nocmnr'),
            'w/o DWT': dict(v='nodwt'),
            'w/o CMNR and DWT': dict(v='none'),
        }
        for name, kw in cfgs.items():
            which = kw.pop('which', ['MC-AE'])
            grs = kw.pop('grs', (True,))
            rs_ = []
            for seed in TEST:
                r, _, _ = run_config(seed, A, S, which=which, grs=grs, **kw)
                rs_.append({(k[0], 'cfg'): v for k, v in r.items()})
            out[name] = tostr(agg(rs_))
            print('abl', name, round(time.time() - t0), flush=True)
        names = ['RMS', 'Var', 'Skew', 'Kurt', 'P2P']
        for drop in range(5):
            keep = [i for i in range(5) if i != drop]
            rs_ = []
            for seed in TEST:
                d = load(seed)
                Xtr, Xn, Xm, ys = make_inputs(d, 'TL', feats=keep)
                torch.manual_seed(seed)
                mdl = P.ProbabilisticFeatureAutoencoder(input_dim=4)
                opt = torch.optim.Adam(mdl.parameters(), lr=0.01)
                Xt = torch.tensor(Xtr, dtype=torch.float32)
                for e in range(150):
                    mdl.train(); opt.zero_grad(); l = ((mdl(Xt) - Xt) ** 2).mean(); l.backward(); opt.step()
                sn = P.mc_scores(mdl, Xn, seed=seed)[0]
                r = {}
                for scn in SC:
                    a, b, thr = decide(sn, P.mc_scores(mdl, Xm[scn], seed=seed)[0], A, S)
                    r[(scn, 'cfg')] = metr(ys[scn], b, thr, a)
                rs_.append(r)
            out[f'w/o feature {names[drop]}'] = tostr(agg(rs_))
        print('feat', round(time.time() - t0), flush=True)
        for M in [1, 5, 10, 30, 50, 100]:
            rs_ = []
            for seed in TEST:
                d = load(seed)
                Xtr, Xn, Xm, ys = make_inputs(d, 'TL')
                m, _ = P.train_ae(Xtr, seed, val_mask=np.r_[VALMASK, VALMASK])
                r = {}
                for scn in ANOM:
                    reps = []
                    for k in range(5):
                        sn = P.mc_scores(m, Xn, M=M, seed=seed + 97 * k)[0]
                        a, b, thr = decide(sn, P.mc_scores(m, Xm[scn], M=M, seed=seed + 97 * k)[0], A, S)
                        reps.append(metr(ys[scn], b, thr, a))
                    r[(scn, 'cfg')] = dict(auprc=float(np.mean([x['auprc'] for x in reps])),
                                           f1=float(np.mean([x['f1'] for x in reps])),
                                           f1_rep_sd=float(np.std([x['f1'] for x in reps])),
                                           auprc_rep_sd=float(np.std([x['auprc'] for x in reps])))
                rs_.append(r)
            out[f'M={M}'] = tostr(agg(rs_))
        for p in [0.0, 0.05, 0.15, 0.3, 0.5]:
            rs_ = []
            for seed in TEST:
                r, _, _ = run_config(seed, A, S, which=['MC-AE'], grs=(True,), p=p)
                rs_.append({(k[0], 'cfg'): v for k, v in r.items()})
            out[f'p={p}'] = tostr(agg(rs_))
        print('M/p', round(time.time() - t0), flush=True)
        json.dump(out, open(f'{R}/ablations.json', 'w'), indent=1)

    if step == 'strength':
        A, S = sel()
        rows = []
        for f in sorted(glob.glob(f'{C}/sweep_*.npz')):
            seed = int(f.split('_')[-1].split('.')[0])
            for mode, oa in (('TL', True), ('ABS', False)):
                d = load(seed, 'sweep')
                Xtr, Xn, Xm, ys = make_inputs(d, mode, oa=oa)
                dets, _, _ = fit_detectors(Xtr, seed, ['Mahalanobis', 'IsolationForest', 'MC-AE'])
                for k, fn in dets.items():
                    sn = fn(Xn)
                    for scn, X in Xm.items():
                        sm = fn(X)
                        for gr in (False, True):
                            a, b, thr = decide(sn, sm, A, S, gr)
                            m = metr(ys[scn], b, thr, a)
                            rows.append(dict(seed=seed, s=float(scn[1:]), method=f"{mode}:{'GR-' if gr else ''}{k}",
                                             pos=float(ys[scn].mean()), **m))
        json.dump(rows, open(f'{R}/strength.json', 'w'))
    print('done', step, round(time.time() - t0))
