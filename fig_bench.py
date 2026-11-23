import sys; sys.path.insert(0,'/home/claude/exp')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import dassim as D
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
C='/home/claude/exp/cache'
b=np.load(f'{C}/gathers_base.npz'); s4=np.load(f'{C}/gathers_S4_plume_nuis.npz')
acq=D.Acquisition(0)
fig=plt.figure(figsize=(6.5,4.6))
gs=fig.add_gridspec(2,3,height_ratios=[0.75,1],hspace=0.45,wspace=0.32)
ax=fig.add_subplot(gs[0,:])
sc=ax.scatter(D.XS,-D.ZS,c=np.arange(D.NS),s=2,cmap='viridis')
ax.plot(acq.src_x,0,'r*',ms=10,label='Hammer source')
th=np.linspace(0,2*np.pi,100)
ax.plot(acq.plume_x+acq.plume_r*np.cos(th),-acq.plume_z+acq.plume_r*np.sin(th),'r--',lw=1,label='Near-well perturbation (S3, S4)')
x5=D.XS[1790]
ax.plot(x5+acq.plume_r*np.cos(th),-acq.plume_z+acq.plume_r*np.sin(th),'m:',lw=1.2,label='Distal perturbation (S5)')
ax.set_xlabel('Along-trench distance (m)'); ax.set_ylabel('Depth (m)')
cb=fig.colorbar(sc,ax=ax,pad=0.01); cb.set_label('Channel index')
ax.set_ylim(-108,8); ax.legend(loc='lower right',fontsize=7,frameon=True,ncol=1); ax.set_title('A',loc='left',fontweight='bold')
t=np.arange(b['raw'].shape[0])*D.DT
ext=[0,D.NS,t[-1],0]
for i,(arr,tt) in enumerate([(b['raw'],'B'),(b['stack'],'C'),(s4['gtdiff'],'D')]):
    a=fig.add_subplot(gs[1,i])
    if tt=='D':
        im=a.imshow(np.log10(arr+1e-3),aspect='auto',cmap='magma',extent=ext,vmin=-2,vmax=2)
        cbar=fig.colorbar(im,ax=a,pad=0.02); cbar.set_label('log$_{10}$ perturbation energy')
    else:
        v=np.percentile(np.abs(arr),99); a.imshow(arr,aspect='auto',cmap='seismic',vmin=-v,vmax=v,extent=ext)
    a.set_xlabel('Channel'); 
    if i==0: a.set_ylabel('Time in window (s)')
    a.set_title(tt,loc='left',fontweight='bold')
plt.savefig('/home/claude/paper/figs/fig2_benchmark.pdf',bbox_inches='tight'); plt.savefig('/tmp/fb.png',dpi=100,bbox_inches='tight')
