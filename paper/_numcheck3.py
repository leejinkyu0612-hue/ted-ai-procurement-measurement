import re
def nums(s):
    s=re.sub(r"\\(cite|ref|label|eqref|orcidID|url)\{[^}]*\}","",s); s=re.sub(r"\\begin\{tikzpicture\}","",s)
    return re.findall(r"(?<![A-Za-z])-?\d+(?:[.,]\d+)*", s)
def norm(x):
    x=x.replace(",","").lstrip("-"); return x
r2=open("R3.tex",encoding="utf-8").read()
pool=open("R2.tex",encoding="utf-8").read()+open("../results/r3_digest.txt",encoding="utf-8").read()+open("R1.tex",encoding="utf-8").read()+open("../results/r1_digest.txt",encoding="utf-8").read()+open("../results/r2_digest.txt",encoding="utf-8").read()
real=open("../results/r1_digest_realizedpi.txt",encoding="utf-8").read()
P=set(); 
for x in nums(pool):
    n=norm(x); P.add(n)
    try:
        f=float(n)
        for d in (0,1,2,3): P.add(f"{round(f,d):.{d}f}"); P.add(f"{round(f*100,d):.{d}f}")
    except: pass
R=set()
for x in nums(real):
    n=norm(x)
    try:
        f=float(n)
        for d in (0,1,2,3): R.add(f"{round(f,d):.{d}f}"); R.add(f"{round(f*100,d):.{d}f}")
    except: pass
bad=[]
for m in re.finditer(r"(?<![A-Za-z])-?\d+(?:[.,]\d+)*", r2):
    n=norm(m.group(0))
    if n not in P:
        ctx=r2[max(0,m.start()-60):m.end()+30].replace("\n"," ")
        bad.append((n, n in R, ctx))
seen=set()
for b in bad:
    if (b[0],b[2]) in seen: continue
    seen.add((b[0],b[2])); print(b[0], "REALIZED" if b[1] else "", "|", b[2])

