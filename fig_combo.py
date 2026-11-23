import sys, json; sys.path.insert(0,'/home/claude/exp')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import analyze2 as A
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
R='/home/claude/exp/results'; al,sg=A.sel()
rows=json.load(open(f'{R}/strength.json'))
meths=[('TL:GR-MC-AE','TL-GR-MCAE (proposed)','#d62728','-'),('TL:MC-AE','TL-MCAE','#ff7f0e','--'),('TL:GR-IsolationForest','TL-GR-IF','#2ca02c','-'),
       ('TL:Mahalanobis','TL-Mahalanobis','#9467bd','--'),('ABS:GR-MC-AE','Original GR-MCAE','#7f7f7f','-')]
ss=sorted({r['s'] for r in rows})
fig,axs=plt.subplots(1,4,figsize=(6.7,2.25),gridspec_kw=dict(width_ratios=[1,1,1.05,1.05]))
for a,met,t in zip(axs[:2],['auprc','f1'],['A  AUPRC','B  F1']):
    for k,lab,c,ls in meths:
        v=np.array([[r[met] for r in rows if r['method']==k and r['s']==s] for s in ss])
        a.errorbar(ss,v.mean(1),yerr=v.std(1),label=lab,color=c,ls=ls,marker='o',ms=2.5,capsize=1.5,lw=1)
    if met=='auprc':
        a.plot(ss,[np.mean([r['pos'] for r in rows if r['s']==s and r['method']==meths[0][0]]) for s in ss],'k:',lw=1)
    a.set_xlabel('Strength $s$'); a.set_title(t,loc='left',fontweight='bold',fontsize=8); a.grid(alpha=0.3)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=5,fontsize=7,frameon=False,bbox_to_anchor=(0.5,-0.06))
g=json.load(open(f'{R}/grid.json'))
alphas=sorted({float(k.split('|')[2]) for k in g}); sigs=sorted({float(k.split('|')[3]) for k in g})
def mat(scns,met): return np.array([[np.mean([g[f'{sd}|{c}|{a}|{s}'][met] for sd in A.TEST for c in scns]) for a in alphas] for s in sigs])
for a,(M,t,cm) in zip(axs[2:],[(mat(A.ANOM,'f1'),'C  F1 (S3–S5)','viridis'),(mat(A.NULL,'flag'),'D  Null FAR','magma_r')]):
    im=a.imshow(M,aspect='auto',cmap=cm,origin='lower')
    a.set_xticks(range(0,len(alphas),2)); a.set_xticklabels([f'{alphas[i]:g}' for i in range(0,len(alphas),2)],fontsize=7,rotation=90)
    a.set_yticks(range(len(sigs))); a.set_yticklabels([f'{x:g}' for x in sigs],fontsize=7)
    a.set_xlabel('$\\alpha$'); a.set_title(t,loc='left',fontweight='bold',fontsize=8)
    a.plot(alphas.index(al),sigs.index(sg),'w*',ms=7,mec='k'); c=plt.colorbar(im,ax=a,pad=0.03); c.ax.tick_params(labelsize=7)
axs[2].set_ylabel('$\\sigma_g$')
plt.tight_layout(w_pad=0.6); plt.savefig('/home/claude/paper/figs/fig5_combo.pdf',bbox_inches='tight'); plt.savefig('/tmp/combo.png',dpi=110,bbox_inches='tight')
