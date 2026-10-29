import json, matplotlib, matplotlib.ticker
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open('final.json'))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.linewidth':0.6})
pal=['#2a78d6','#eb6834','#1baf7a','#e87ba4','#eda100','#008300']
mk=['o','s','^','D','v','P']
fig,ax=plt.subplots(1,2,figsize=(7.2,3.9),dpi=600)
h1=[1/N for N in [50,100,200,400,800]]; h2=[1/N for N in [10,20,40,80,160,320]]
s1=[('RK2','RK2'),('Verlet','Verlet (Störmer)'),('OptExp','Optimal explicit, $m=4,k=6$'),('OptImp','Optimal implicit, $m=4,k=6$'),('RK4','RK4'),('Numerov','Numerov')]
s2=[('E53','Optimal explicit, $m=5,k=3$'),('E64','Optimal explicit, $m=6,k=4$'),('RK4','RK4'),('I63','Optimal implicit, $m=6,k=3$')]
for i,(k,l) in enumerate(s1):
    ax[0].loglog(h1,R['T1'][k],marker=mk[i],ms=4,lw=1.2,color=pal[i],label=l)
for i,(k,l) in enumerate(s2):
    c2={'E53':('#6250d6','X'),'E64':('#e34948','*'),'RK4':(pal[4],mk[4]),'I63':(pal[5],mk[5])}[k]; ax[1].loglog(h2,R['T2'][k],marker=c2[1],ms=4.5,lw=1.2,color=c2[0],label=l)
for a,t in zip(ax,['(a)','(b)']):
    a.set_xlabel('Step size $h$'); a.grid(True,which='major',color='#e6e6e3',lw=0.5); a.set_axisbelow(True)
    a.set_title(t,loc='left',fontsize=9)
    for s in ('top','right'): a.spines[s].set_visible(False)
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
import matplotlib.ticker as mt
ax[0].set_xticks(h1); ax[0].set_xticklabels(['1/50','1/100','1/200','1/400','1/800'])
ax[1].set_xticks(h2); ax[1].set_xticklabels(['1/10','1/20','1/40','1/80','1/160','1/320'],fontsize=7)
ax[0].set_ylabel('Maximum absolute error')
for a in ax: a.legend(fontsize=6.5,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=2)
fig.tight_layout(); fig.savefig('figure1.png',dpi=600); fig.savefig('figure1.pdf')
