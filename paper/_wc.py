import re
t=open('R1.tex',encoding='utf-8').read()
secs=re.split(r'(\\section\{[^}]*\})',t)
for i in range(1,len(secs),2): print(secs[i], len(secs[i+1].split()))
a=re.search(r'\\begin\{abstract\}(.*?)\\keywords',t,re.S).group(1); print('abstract words',len(a.split()))
