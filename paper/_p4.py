import sys,re
s=open('claude_R3.tex',encoding='utf-8').read()
def rep(a,b,n=1):
    global s
    c=s.count(a)
    if c!=n: print(f'WARN count {c}!={n}: {a[:90]}'); return
    s=s.replace(a,b)
rep(r"Two LLMs (Claude Haiku 5.5 and Gemini 2.5 Flash-Lite, accessed through OpenRouter) were used as research instruments to annotate the reference sample, as described in Sect.~3.2.",
    r"Three LLMs (Claude Haiku 5.5, Gemini 2.5 Flash-Lite, and DeepSeek-Chat-v3.1, accessed through OpenRouter) were used as research instruments for reference annotation and sensitivity analysis, as described in Sect.~3.2.")
open('R3.tex','w',encoding='utf-8').write(s)
for pat in [r'refit', r'CPV division', r'currenc', r'Holm']:
    for m in re.finditer(pat, s): print('>>', pat, '|', s[max(0,m.start()-160):m.end()+160].replace('\n',' '))
