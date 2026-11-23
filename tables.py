import json, numpy as np
R='/home/claude/exp/results'
a=json.load(open(f'{R}/main_agg.json'))
def g(s,m,k,sd=False,fmt='{:.3f}'):
    v=a.get(f'{s}|{m}',{}).get(k)
    if v is None: return '--'
    return (fmt.format(v[0])+(' $\\pm$ '+fmt.format(v[1]) if sd else ''))
SC3=['S3_plume','S4_plume_nuis','S5_distal_nuis']
def far(m): return g('S1_repeat',m,'flag'), g('S2_nuisance',m,'flag')
rows=[('ABS:GR-MC-AE','Original GR-MCAE~\\cite{Gal2016} (absolute features)'),
      ('4D-dRMS','4D RMS difference'),('GR-4D-dRMS','GR-4D RMS difference'),
      ('Mahalanobis','TL-Mahalanobis'),('PCA-Recon','TL-PCA'),('IsolationForest','TL-IF'),('OC-SVM','TL-OC-SVM'),('LOF','TL-LOF'),('Det-AE','TL-AE (deterministic)'),('MC-AE','TL-MCAE'),
      ('GR-Mahalanobis','TL-GR-Mahalanobis'),('GR-IsolationForest','TL-GR-IF'),('GR-OC-SVM','TL-GR-OC-SVM'),('GR-MC-AE','\\textbf{TL-GR-MCAE (proposed)}')]
print('=== MAIN')
for k,n in rows:
    print(n,'&',' & '.join(far(k)),'&',' & '.join(f"{g(s,k,'auprc')} & {g(s,k,'f1')}" for s in SC3),'\\\\')
print('AUROC'); 
for k,n in rows: print(n, [g(s,k,'auroc') for s in SC3])
print('prox', [g(s,'SourceProximity','auroc') for s in SC3],[g(s,'SourceProximity','auprc') for s in SC3])
print('=== GR effect (TL and ABS) mean over S3-S5 AUPRC, F1, null FAR')
for d in ['Mahalanobis','PCA-Recon','IsolationForest','OC-SVM','LOF','Det-AE','MC-AE']:
    out=[]
    for pre in ['ABS:','']:
        for gr in ['','GR-']:
            k=pre+gr+d
            ap=np.mean([a[f'{s}|{k}']['auprc'][0] for s in SC3]); f1=np.mean([a[f'{s}|{k}']['f1'][0] for s in SC3]); fa=np.mean([a[f'{s}|{k}']['flag'][0] for s in ['S1_repeat','S2_nuisance']])
            mo=np.mean([a[f'{s}|{k}']['moran'][0] for s in SC3])
            out.append(f'{ap:.3f}/{f1:.3f}/{fa:.3f}/{mo:.2f}')
    print(d,out)
w=json.load(open(f'{R}/wilcoxon.json'))
print('=== WILCOXON')
for k,v in w.items(): print(k, round(v[0],3), f'{v[1]:.2e}')
u=json.load(open(f'{R}/uncertainty.json'))
print('=== UNC')
for s in ['S1_repeat','S2_nuisance']+SC3:
    e=[x for x in u if x['scn']==s]
    print(s, 'rho',np.mean([x['rho'] for x in e]).round(3), 'aurocvar',np.nanmean([x.get('auroc_var',np.nan) for x in e]).round(3),'aurocmu',np.nanmean([x.get('auroc_mu',np.nan) for x in e]).round(3),
          'var_tp',np.nanmedian([x.get('var_tp',np.nan) for x in e]),'var_fp',np.nanmedian([x.get('var_fp',np.nan) for x in e]),'flag',np.nanmedian([x['var_flag'] for x in e]),'unflag',np.nanmedian([x['var_unflag'] for x in e]))
print(json.load(open(f'{R}/infer_time.json')))
try:
    ab=json.load(open(f'{R}/ablations.json'))
    print('=== ABL')
    for k,v in ab.items():
        def m(sc,met):
            x=[v.get(f'{s}|cfg',{}).get(met) for s in sc]; x=[y[0] for y in x if y]
            return np.mean(x) if x else np.nan
        print(f'{k:42s}', 'AUPRC %.3f F1 %.3f FAR %.3f Moran %.2f | f1sd %.3f apsd %.3f'%(m(SC3,'auprc'),m(SC3,'f1'),m(['S1_repeat','S2_nuisance'],'flag'),m(SC3,'moran'),m(SC3,'f1_rep_sd'),m(SC3,'auprc_rep_sd')))
except Exception as e: print(e)
