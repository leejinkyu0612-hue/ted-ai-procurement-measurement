import re,sys
s=open('R2.tex',encoding='utf-8').read()
def rep(a,b,n=1):
    global s
    c=s.count(a)
    if c!=n: sys.exit(f'count {c}!={n}: {a[:80]}')
    s=s.replace(a,b)
m=re.search(r'Inclusion probabilities are \$\\pi_h=n_h/N_h\$, where \$n_h\$ counts \\emph\{labelled\} notices;.*?\.', s, re.S)
print('OLD:', m.group(0)[:400])
# find the full sentence(s) up to next sentence about pi_C
seg=s[m.start():m.start()+600]; print(seg)
open('_seg.txt','w',encoding='utf-8').write(seg)
