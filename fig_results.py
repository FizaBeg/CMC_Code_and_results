import sys, json; sys.path.insert(0,'/home/claude/exp')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import pipeline as P, analyze2 as A
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
FIG='/home/claude/paper/figs'; R='/home/claude/exp/results'
al,sg=A.sel()
which=sys.argv[1:]
tw=(P.T_STARTS+P.T_WIN/2)*P.DT; cw=P.S_STARTS+P.S_WIN/2
ext=[cw[0],cw[-1],tw[-1],tw[0]]
def show(ax,v,title,cmap='viridis',lab=None,y=None,vmin=None,vmax=None,cb=''):
    im=ax.imshow(v.reshape(P.NTW,P.NSW),aspect='auto',cmap=cmap,extent=ext,vmin=vmin,vmax=vmax,interpolation='nearest')
    if y is not None: ax.contour(cw,tw,y.reshape(P.NTW,P.NSW).astype(float),[0.5],colors='r',linewidths=0.8)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=8)
    c=plt.colorbar(im,ax=ax,pad=0.02); c.set_label(cb,fontsize=7); c.ax.tick_params(labelsize=7)
    return im
if 'maps' in which:
    seed=0; d=A.load(seed); scn='S4_plume_nuis'
    # original (ABS, no OA)
    Xtr,Xn,Xm,ys=A.make_inputs(d,'ABS',oa=False); dets,_,_=A.fit_detectors(Xtr,seed,['MC-AE'])
    an,b_abs,thr_abs=A.decide(dets['MC-AE'](Xn),dets['MC-AE'](Xm[scn]),al,sg)
    Xtr,Xn,Xm,ys=A.make_inputs(d,'TL'); dets,model,_=A.fit_detectors(Xtr,seed,['MC-AE'])
    mn,vn=P.mc_scores(model,Xn,seed=seed); mu,var=P.mc_scores(model,Xm[scn],seed=seed)
    _,b_raw,thr_raw=A.decide(mn,mu,al,sg,gr=False); _,b_gr,thr_gr=A.decide(mn,mu,al,sg)
    y=ys[scn]
    fig,axs=plt.subplots(2,3,figsize=(6.5,4.4)); axs=axs.ravel()
    show(axs[0],d[f'{scn}_ratio'] if False else y.astype(float),'A  Ground truth',cmap='Greys',cb='label')
    show(axs[1],b_abs,'B  Original (absolute)',y=d[f'{scn}_y'],cb='std. score')
    show(axs[2],b_raw,'C  TL-MCAE (no graph)',y=y,cb='std. score',vmin=np.percentile(b_raw,1),vmax=np.percentile(b_raw,99.5))
    show(axs[3],b_gr,'D  TL-GR-MCAE (proposed)',y=y,cb='std. score')
    show(axs[4],(b_gr>thr_gr).astype(float),'E  Decision (P95 rule)',cmap='Greys',y=y,cb='flag')
    show(axs[5],np.log10(var+1e-6),'F  MC Dropout variance',cmap='magma',y=y,cb='log$_{10}\\,\\sigma^2$')
    for i,a in enumerate(axs):
        a.tick_params(labelsize=7)
        if i%3==0: a.set_ylabel('Time (s)')
        if i>=3: a.set_xlabel('Channel')
    plt.tight_layout(); plt.savefig(f'{FIG}/fig4_maps.pdf',bbox_inches='tight'); plt.savefig('/tmp/maps.png',dpi=100,bbox_inches='tight')
if 'loss' in which:
    h=np.load(f'{R}/hists.npy')  # seeds x epochs x 2
    fig,ax=plt.subplots(figsize=(4.2,2.6))
    ep=np.arange(1,h.shape[1]+1)
    for j,(lab,c) in enumerate([('Training','#1f77b4'),('Spatially held-out validation','#d62728')]):
        m=h[:,:,j].mean(0); s=h[:,:,j].std(0)
        ax.plot(ep,m,color=c,label=lab); ax.fill_between(ep,m-s,m+s,color=c,alpha=0.2)
    ax.set_yscale('log'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE loss'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{FIG}/fig3_loss.pdf',bbox_inches='tight'); plt.savefig('/tmp/loss.png',dpi=100)
if 'sens' in which:
    g=json.load(open(f'{R}/grid.json'))
    alphas=sorted({float(k.split('|')[2]) for k in g}); sigs=sorted({float(k.split('|')[3]) for k in g})
    def mat(scns,met):
        M=np.zeros((len(sigs),len(alphas)))
        for i,s in enumerate(sigs):
            for j,a in enumerate(alphas):
                M[i,j]=np.mean([g[f'{sd}|{c}|{a}|{s}'][met] for sd in A.TEST for c in scns])
        return M
    fig,axs=plt.subplots(1,3,figsize=(6.6,2.5))
    for a,(M,t,cm) in zip(axs,[(mat(A.ANOM,'auprc'),'A  AUPRC (S3–S5)','viridis'),(mat(A.ANOM,'f1'),'B  F1 (S3–S5)','viridis'),(mat(A.NULL,'flag'),'C  False-alarm rate (S1–S2)','magma_r')]):
        im=a.imshow(M,aspect='auto',cmap=cm,origin='lower')
        a.set_xticks(range(len(alphas))); a.set_xticklabels([f'{x:g}' for x in alphas],rotation=90,fontsize=7)
        a.set_yticks(range(len(sigs))); a.set_yticklabels([f'{x:g}' for x in sigs],fontsize=7)
        a.set_xlabel('$\\alpha$'); a.set_title(t,loc='left',fontweight='bold',fontsize=8)
        j=alphas.index(al); i=sigs.index(sg); a.plot(j,i,'w*',ms=8,mec='k')
        c=plt.colorbar(im,ax=a,pad=0.02); c.ax.tick_params(labelsize=7)
    axs[0].set_ylabel('$\\sigma_g$')
    plt.tight_layout(); plt.savefig(f'{FIG}/fig6_sensitivity.pdf',bbox_inches='tight'); plt.savefig('/tmp/sens.png',dpi=100)
if 'strength' in which:
    rows=json.load(open(f'{R}/strength.json'))
    meths=[('TL:GR-MC-AE','TL-GR-MCAE (proposed)','#d62728','-'),('TL:MC-AE','TL-MCAE','#ff7f0e','--'),('TL:GR-IsolationForest','TL-GR-IF','#2ca02c','-'),
           ('TL:Mahalanobis','TL-Mahalanobis','#9467bd','--'),('ABS:GR-MC-AE','Original GR-MCAE','#7f7f7f','-')]
    ss=sorted({r['s'] for r in rows})
    fig,axs=plt.subplots(1,2,figsize=(6.4,2.5))
    for a,met,t in zip(axs,['auprc','f1'],['A  AUPRC','B  F1 at P95 rule']):
        for k,lab,c,ls in meths:
            v=np.array([[r[met] for r in rows if r['method']==k and r['s']==s] for s in ss])
            a.errorbar(ss,v.mean(1),yerr=v.std(1),label=lab,color=c,ls=ls,marker='o',ms=3,capsize=2,lw=1)
        if met=='auprc':
            pos=[np.mean([r['pos'] for r in rows if r['s']==s and r['method']==meths[0][0]]) for s in ss]
            a.plot(ss,pos,'k:',lw=1,label='Chance (prevalence)')
        a.set_xlabel('Perturbation strength $s$'); a.set_title(t,loc='left',fontweight='bold',fontsize=8); a.grid(alpha=0.3)
    axs[1].legend(fontsize=6.5,loc='upper left')
    plt.tight_layout(); plt.savefig(f'{FIG}/fig5_strength.pdf',bbox_inches='tight'); plt.savefig('/tmp/str.png',dpi=100)
