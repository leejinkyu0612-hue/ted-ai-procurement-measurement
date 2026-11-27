import json, re
s=open('S17.tex',encoding='utf-8').read(); orig=s
fx=json.load(open('s18_fixes.json',encoding='utf-8'))
log=[]
for i,x in enumerate(fx,1):
    old,new=x['old'],x['new']
    if i==1: new=new.replace('Multilingual EU Procurement','Multilingual European Procurement')
    if i==4: new=new.replace('for EU procurement','for European procurement')
    if i==20: new='covered the true coefficient (0.10) in only 85.5\\% of replications.'
    if i==25 and '\\label{sec:ref}' not in s: log.append((i,'skip: no label')); continue
    if i==22 and '\\label{eq:schur}' not in s: new=new.replace(' (Eq.~\\eqref{eq:schur})','')
    c=s.count(old)
    if c!=1: log.append((i,f'skip count {c}')); continue
    s=s.replace(old,new); log.append((i,'applied'))
# running title consistent with new title
s=s.replace(r'\titlerunning{Identifying AI-Related Notices in Multilingual Procurement}', r'\titlerunning{Identifying AI-Related Notices in Multilingual Procurement}')
for a,b in [('unlabelled','unlabeled'),('Unlabelled','Unlabeled'),('labelled','labeled'),('Labelled','Labeled'),('labelling','labeling')]:
    s=s.replace(a,b)
def nums(t):
    t=re.sub(r"\\(cite|ref|label|eqref|orcidID|url)\{[^}]*\}","",t); return re.findall(r"\d+(?:[.,]\d+)*",t)
from collections import Counter
d=Counter(nums(s))-Counter(nums(orig)); d2=Counter(nums(orig))-Counter(nums(s))
open('S18.tex','w',encoding='utf-8').write(s)
print(log); print('added numbers', dict(d)); print('removed numbers', dict(d2))
