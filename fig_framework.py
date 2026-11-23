import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'mathtext.fontset':'dejavusans'})
fig,ax=plt.subplots(figsize=(6.5,3.9)); ax.set_xlim(0,130); ax.set_ylim(0,78); ax.axis('off')
def box(x,y,w,h,t,fc):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.3,rounding_size=1.5',fc=fc,ec='#333',lw=0.8))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
def arr(x1,y1,x2,y2,c='#333',ls='-'):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',lw=1.0,color=c,ls=ls))
rows=[('A','Signal conditioning','#dbe9f6',['Raw DAS\nshots','CMNR\n(median)','DWT db4\ndenoising','Onset\nalignment','Stacking\n(full / half)']),
      ('B','Time-lapse representation','#e6dff2',['Windows\n250 × 56','Five\nmoments','Residual\n$\\mathbf{z}^{mon}-\\mathbf{z}^{base}$','Robust\nre-centring']),
      ('C','Scoring and decision','#dcefd9',['MC-Dropout\nAE (M = 30)','Graph\ndiffusion','Standardise\n+ P95 rule','Anomaly and\nvariability maps'])]
y0=[56,31,6]; W=21; H=13
for (lab,title,c,items),y in zip(rows,y0):
    ax.add_patch(FancyBboxPatch((0.5,y-3.5),129,H+9,boxstyle='round,pad=0.2,rounding_size=2',fc=c,ec='none',alpha=0.6))
    ax.text(128,y+H+3.3,f'({lab}) {title}',fontsize=8,fontweight='bold',va='center',ha='right')
    n=len(items); gap=(126-n*W)/(n-1) if n>1 else 0
    xs=[2+i*(W+gap) for i in range(n)]
    for x,t in zip(xs,items): box(x,y,W,H,t,'white')
    for i in range(n-1): arr(xs[i]+W+0.6,y+H/2,xs[i+1]-0.6,y+H/2)
# connectors between rows
ax.plot([117.5,117.5,12.5],[56-0.5,50.6,50.6],color='#333',lw=1.0); arr(12.5,50.6,12.5,31+H+0.6)
ax.plot([117.5,117.5,12.5],[31-0.5,25.6,25.6],color='#333',lw=1.0); arr(12.5,25.6,12.5,6+H+0.6)
plt.savefig('/home/claude/paper/figs/fig1_framework.pdf',bbox_inches='tight'); plt.savefig('/tmp/fw.png',dpi=110,bbox_inches='tight')
