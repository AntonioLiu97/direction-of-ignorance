"""Shared all-model layout and Figure-4-compatible paired Wald SE."""
import numpy as np
POSITIONS={'llama32_1b':(0,0),'llama32_3b':(1,0),'llama31_8b':(2,0),
 'gemma3_1b':(0,1),'gemma3_4b':(1,1),'qwen_1p5b':(0,2),
 'qwen_math_1p5b':(1,2),'qwen_r1_1p5b':(2,2),'qwen_3b':(0,3)}
def wald(row):
 n=row['n']; wrong=n-row['baseline_correct']; mean=row['difference']
 assert wrong>0
 assert row['corrected']-row['broken']==row['tuned_correct']-row['baseline_correct']
 se=np.sqrt(max((row['corrected']+row['broken'])/n-mean**2,0)/n)
 return 100*n/wrong*mean,100*n/wrong*se
def draw(ax,rows,**kwargs):
 lookup={round(r['alpha'],2):r for r in rows if .4<=r['alpha']<=1.2}
 x=np.arange(8,25)/20
 y=[wald(lookup[round(a,2)])[0] if round(a,2) in lookup else np.nan for a in x]
 e=[wald(lookup[round(a,2)])[1] if round(a,2) in lookup else np.nan for a in x]
 return ax.errorbar(x,y,yerr=e,elinewidth=1,capsize=2,capthick=1,**kwargs)
def baseline_subtitle(fitted,control=None):
 p=100*fitted[0]['baseline_correct']/fitted[0]['n']
 if control:
  q=100*control[0]['baseline_correct']/control[0]['n']
  if abs(p-q)>1e-9:return f'Baseline: prior {p:.2f}%; random {q:.2f}%'
 return f'Baseline accuracy: {p:.2f}%'
def limits(groups):
 high=max((sum(wald(r)) for rows in groups for r in rows if .4<=r['alpha']<=1.2),default=20)
 return -20,max(20,5*np.ceil(high/5))
