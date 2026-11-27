import re
s=open('S12.tex',encoding='utf-8').read()
for m in re.finditer(r'\\begin\{figure\}.*?\\end\{figure\}', s, re.S):
    f=m.group(0); cap=re.search(r'\\caption\{(.*)\}\\label\{([^}]*)\}', f, re.S)
    print(len(f), cap.group(2) if cap else None); print(cap.group(1)[:600] if cap else f[-600:]); print('---')
