import re,sys
s=open('R2.tex',encoding='utf-8').read()
def show(pat):
    for m in re.finditer(pat,s): print('>>', s[max(0,m.start()-150):m.end()+150].replace('\n',' '))
show(r'\\pi_h=n_h/N_h'); show(r'1\.52 points'); show(r'in only'); show(r'differ slightly from')
