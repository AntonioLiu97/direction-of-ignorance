"""Versioned, conservative supplementary scoring. Never modifies saved predictions."""
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'unit_rescore_v1'
NUM=r'[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
# Exact conversions only; no months/years, currencies, or ambiguous volume units.
UNITS={}
for dimension, entries in {
    'time':[(1,'second seconds sec secs'),(60,'minute minutes min mins'),(3600,'hour hours hr hrs'),(86400,'day days'),(604800,'week weeks')],
    'length':[(1,'meter meters metre metres'),('.01','centimeter centimeters centimetre centimetres cm'),('.001','millimeter millimeters mm'),(1000,'kilometer kilometers kilometre kilometres km'),('.0254','inch inches'),('.3048','foot feet ft'),('.9144','yard yards'),('1609.344','mile miles')],
    'mass':[(1,'gram grams g'),(1000,'kilogram kilograms kg'),('.001','milligram milligrams mg')],
    'volume':[(1,'liter liters litre litres'),('.001','milliliter milliliters ml')],
}.items():
    for factor, aliases in entries:
        for name in aliases.split(): UNITS[name]=(dimension,Decimal(str(factor)))
UNIT='|'.join(sorted(UNITS,key=len,reverse=True))
QTY=re.compile(r'('+NUM+r')\s*('+UNIT+r')\b',re.I)

def number(s):
    s=str(s).strip().replace(',','').replace('$','')
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?',s): return None
    try: return Decimal(s)
    except InvalidOperation: return None

def associated_unit(text,value):
    # The last explicit quantity must agree with the extracted answer. Never
    # search backward past a different final quantity to rescue an answer.
    text=re.sub(r'<<.*?>>','',text)
    matches=list(QTY.finditer(text))
    if not matches: return None
    m=matches[-1]
    if number(m[1])!=value or len(text)-m.end()>160: return None
    return m[2].lower()

def requested_unit(question):
    m=re.search(r'how\s+many\s+('+UNIT+r')\b',question,re.I)
    return m[1].lower() if m else None

def score(row,doc):
    old=row['scores']['flexible-extract']; value=number(old['answer'])
    target=number(doc['answer'].split('####')[-1])
    reference=doc['answer'].split('####')[0]
    demand=requested_unit(doc['question'])
    gu=demand or associated_unit(reference,target)
    pu=associated_unit(row['response'],value)
    numeric=value is not None and target is not None and value==target
    equivalent=False
    if value is not None and target is not None and pu and gu:
        dp,fp=UNITS[pu]; dg,fg=UNITS[gu]
        equivalent=dp==dg and value*fp==target*fg
    # Preserve benchmark decisions; supplement only formatting and verified units.
    # Explicit conversions must not override an explicitly requested unit.
    accepted_units=equivalent and (not demand or UNITS[pu]==UNITS[demand])
    final=bool(old['correct'] or numeric or accepted_units)
    candidates=[number(m.group()) for m in list(re.finditer(NUM,row['response']))[-2:]]
    return dict(original=bool(old['correct']),numeric_normalized=bool(old['correct'] or numeric),
        unit_aware=final,physical_equivalence=bool(old['correct'] or numeric or equivalent),
        last_two_numeric=target is not None and target in candidates,
        extracted=old['answer'],target=str(target),predicted_unit=pu,reference_unit=gu,requested_unit=demand,
        reason='original' if old['correct'] else 'numeric_equivalence' if numeric else 'unit_equivalence' if accepted_units else 'unresolved_or_wrong')

def tests():
    assert number('$2.00')==number('2')
    assert number('2,000.00')==number('2000')
    assert number('-2')!=number('2')
    def case(response,answer,q='How much time?',gold='390 minutes\n#### 390'):
        return score(dict(response=response,scores={'flexible-extract':dict(answer=answer,correct=0)}),dict(question=q,answer=gold))
    assert case('390 minutes. 390 / 60 = 6.5 hours. The final answer is 6.5','6.5')['unit_aware']
    assert not case('390 minutes. 390 / 60 = 6.5 hours. The final answer is 6.5','6.5','How many minutes?')['unit_aware']
    assert not case('390 minutes. The final answer is 5','5')['unit_aware']
    assert not case('6.5 miles. The final answer is 6.5','6.5')['unit_aware']
    assert not case('6.5 months. The final answer is 6.5','6.5')['unit_aware']
    assert case('The final answer is $2.00','$2.00',gold='2 dollars\n#### 2')['unit_aware']

def read(path): return json.loads(path.read_text())
def arm(root,alpha,examples):
    rows=sorted([json.loads(s) for s in (root/f'alpha_{alpha:g}'/'predictions.jsonl').read_text().splitlines()],key=lambda r:r['doc_id'])
    assert len(rows)==len(examples)
    result=[]
    for r,d in zip(rows,examples):
        assert r['question_sha256']==d['question_sha256']
        result.append(dict(train_index=d['train_index'],question_sha256=r['question_sha256'],prompt_sha256=r['prompt_sha256'],
            score=score(r,d),question=d['question'],response=r['response'],gold=d['answer']))
    return result

def paired(base,tuned,metric):
    assert [(x['question_sha256'],x['prompt_sha256']) for x in base]==[(x['question_sha256'],x['prompt_sha256']) for x in tuned]
    x=np.array([r['score'][metric] for r in base],dtype=int); y=np.array([r['score'][metric] for r in tuned],dtype=int)
    diff=y-x; n=len(x); rng=np.random.default_rng(42)
    means=np.concatenate([diff[rng.integers(0,n,(100,n))].mean(1) for _ in range(100)])
    win=int((diff==1).sum()); loss=int((diff==-1).sum())
    return dict(n=n,baseline_correct=int(x.sum()),tuned_correct=int(y.sum()),difference=float(diff.mean()),
        ci95=np.quantile(means,[.025,.975]).tolist(),corrected=win,broken=loss,
        p=float(binomtest(win,win+loss,.5).pvalue) if win+loss else 1.)

def main():
    tests(); OUT.mkdir(exist_ok=True)
    source=ROOT/'train_refinement_1024_v1'; examples=read(source/'manifest.json')['examples']
    sweep={a:arm(source/'3b',a,examples) for a in [i/20 for i in range(25)]}
    held=ROOT/'heldout_3b_1024_v1'; seq=ROOT/'sequential_3b_validation_v1'
    sm=read(seq/'manifest.json')['examples']
    validation={a:arm(held,a,read(held/'manifest.json')['examples']) for a in [1.,.65]}
    for i in range(3):
        for a in validation: validation[a]+=arm(seq/f'block_{i:02}',a,sm[i*1024:(i+1)*1024])
    metrics=['original','numeric_normalized','unit_aware','physical_equivalence','last_two_numeric']
    report={m:dict(sweep=[dict(alpha=a,**paired(sweep[1.],rows,m)) for a,rows in sweep.items()],
        validation=paired(validation[1.],validation[.65],m)) for m in metrics}
    changes=[]
    for group,arms in [('sweep',sweep),('validation',validation)]:
        for a,rows in arms.items():
            for r in rows:
                if r['score']['original']!=r['score']['unit_aware']:
                    changes.append(dict(group=group,alpha=a,**r))
    (OUT/'changed_examples.json').write_text(json.dumps(changes,indent=2)+'\n')
    (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({m:dict(validation=r['validation'],at_065=next(s for s in r['sweep'] if s['alpha']==.65),best=max(r['sweep'],key=lambda s:s['tuned_correct'])) for m,r in report.items()},indent=2))

if __name__=='__main__': main()
