import sys,glob,h5py,json,itertools,numpy as np; sys.path.insert(0,'/home/claude/exp'); import pipeline as P
OUT='/home/claude/field/out'; DT=2e-4
st=json.load(open(f'{OUT}/strikes.json'))
DATES=['20250724','20250726','20250728']
def rz(r):
    med=np.median(r,0); return (r-med)/(1.4826*np.median(np.abs(r-med),0)+1e-9)
def slog(x): return np.sign(x)*np.log1p(np.abs(x))
def rs(s):
    med=np.median(s); return (s-med)/(1.4826*np.median(np.abs(s-med))+1e-12)
def dice(a,b): return 2*(a&b).sum()/max(a.sum()+b.sum(),1)
W={d:[] for d in DATES}; info={d:[] for d in DATES}
for key,v in st.items():
    date,name=key.split('/')
    f=glob.glob(f'raw/*/*/{date}/dphi/{name}')[0]
    with h5py.File(f) as h: X=h['data'][:,:].astype(np.float32)*float(h['header/dataScale'][()])
    for t,hgt in v['strikes']:
        if hgt<1.0 or t<0.3 or t>9.3: continue
        a=int(round((t-0.3)/DT))
        W[date].append(P.dwt_denoise(P.cmnr(X[a:a+5000]))); info[date].append((name,round(t,2)))
    del X
print({d:len(W[d]) for d in DATES}, flush=True)
G15=P.GraphDiffuser(P.window_coords(),sigma_g=15.0); G45=P.GraphDiffuser(P.window_coords(),sigma_g=45.0)
B=W[DATES[0]]; ref=P.envelope_trace(np.mean(B,0))
align=lambda S:[np.roll(y,-P.estimate_shift(P.envelope_trace(y),ref,600),0) for y in S]
B=align(B); nb=len(B); idx=np.arange(nb)
A0,B0=idx[0::2],idx[1::2]
feat=lambda S,ix: P.window_features(np.mean([S[i] for i in ix],0))
rng=np.random.default_rng(0)
alt=[]
while len(alt)<10:
    p=rng.permutation(nb); a_,b_=np.sort(p[:nb//2]),np.sort(p[nb//2:])
    if set(a_)!=set(A0) and set(a_)!=set(B0): alt.append((a_,b_))
res={'n_strikes':{d:len(W[d]) for d in DATES},'strikes_used':info}
Fb=feat(B,idx); FA=feat(B,A0); FB=feat(B,B0)
# ---------- original configuration (absolute features, alpha .05 sigma 45) on strike windows
mu,sd=Fb.mean(0),Fb.std(0)+1e-12; Xb=(Fb-mu)/sd
m,h=P.train_ae(Xb,0); sb,vb=P.mc_scores(m,Xb); thr=np.percentile(G45(sb,0.05),95)
o={'null_far_alt_splits':float(np.mean([(G45(P.mc_scores(m,(feat(B,a_)-mu)/sd)[0],0.05)>thr).mean() for a_,b_ in alt]))}
# ---------- TL-GR-MCAE
m2,s2=np.r_[FA,FB].mean(0),np.r_[FA,FB].std(0)+1e-12; z=lambda F:(F-m2)/s2
r0=slog(rz(z(FA)-z(FB))); model,_=P.train_ae(np.r_[r0,-r0],0)
sn,vn=P.mc_scores(model,r0); thr2=np.percentile(G15(rs(sn),0.1),95)
t={'null_far_alt_splits':float(np.mean([(G15(rs(P.mc_scores(model,slog(rz(z(feat(B,a_))-z(feat(B,b_)))))[0]),0.1)>thr2).mean() for a_,b_ in alt]))}
maps={}
for date in DATES[1:]:
    S=align(W[date]); n=len(S); ix=np.arange(n)
    def tl(ixs):
        r=slog(rz(z(feat(S,ixs))-z(Fb))); mu_,var_=P.mc_scores(model,r); return G15(rs(mu_),0.1),mu_,var_
    def ab(ixs):
        s,v=P.mc_scores(m,(feat(S,ixs)-mu)/sd); return G45(s,0.05),s,v
    for name,fn,th,dd in [('orig',ab,thr,o),('tl',tl,thr2,t)]:
        g,s,v=fn(ix); fl=g>th
        d1=[];
        for k in range(10):
            p=rng.permutation(n); f1=fn(np.sort(p[:n//2]))[0]>th; f2=fn(np.sort(p[n//2:]))[0]>th; d1.append(dice(f1,f2))
        cols=fl.reshape(P.NTW,P.NSW).sum(0)
        dd[date]=dict(flag_rate=float(fl.mean()),dice_split_mean=float(np.mean(d1)),dice_split_sd=float(np.std(d1)),moran=float(P.morans_i(g)),
                      top_channels=[int(P.S_STARTS[c]+28) for c in np.argsort(cols)[::-1][:6] if cols[c]>0],
                      score_median=float(np.median(s)),var_median=float(np.median(v)))
        maps[f'{name}_{date}']=g; maps[f'{name}_{date}_flag']=fl; maps[f'{name}_{date}_var']=v
res['original_config_strike_windows']=o; res['tl_gr_mcae']=t
json.dump(res,open(f'{OUT}/field2_results.json','w'),indent=1)
np.savez_compressed(f'{OUT}/field2_maps.npz',**maps)
np.save(f'{OUT}/stack_base.npy',np.mean(B,0)[:, ::2]); np.save(f'{OUT}/stack_0726.npy',np.mean(align(W['20250726']),0)[:, ::2])
print(json.dumps({k:v for k,v in res.items() if k!='strikes_used'},indent=1))
