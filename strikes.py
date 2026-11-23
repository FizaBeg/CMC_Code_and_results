import sys,glob,h5py,json,numpy as np; sys.path.insert(0,'/home/claude/exp'); import pipeline as P
from scipy.signal import find_peaks
OUT='/home/claude/field/out'
fs=sorted(glob.glob('raw/*/*/*/dphi/*.hdf5'))
res={}
for f in fs:
    with h5py.File(f) as h: X=h['data'][:,:].astype(np.float32)*float(h['header/dataScale'][()])
    Y=P.cmnr(X); del X
    B=50  # 10 ms blocks
    E=np.sqrt((Y.reshape(-1,B,Y.shape[1])**2).mean(1))
    E=E/np.median(E,0,keepdims=True)
    sc=np.percentile(np.log10(E),95,axis=1)
    pk,pr=find_peaks(sc,height=0.35,distance=150)
    on=[]
    for p in pk:
        # onset = first block before peak where score exceeds half the peak height
        h=sc[p]; q=p
        while q>0 and sc[q-1]>0.5*h: q-=1
        on.append((q*B*2e-4, float(h)))
    date=f.split('/')[-3]; name=f.split('/')[-1]
    fixed_hit=any(6.0<=t<=6.9 for t,_ in on)
    res[f'{date}/{name}']=dict(strikes=on, fixed_window_contains_strike=fixed_hit)
    print(date,name,[(round(t,2),round(h,2)) for t,h in on], 'fixed6-7:',fixed_hit, flush=True)
json.dump(res,open(f'{OUT}/strikes.json','w'),indent=1)
