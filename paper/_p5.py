import re
s=open('R3.tex',encoding='utf-8').read()
for pat in [r'The analysis plan', r'\\keywords\{[^}]*\}', r'Use of generative AI\.\}[^\n]*\n[^\n]*', r'\\section\{Conclusion\}\n[^\n]*', r'The contribution is']:
    for m in re.finditer(pat, s): print('>>', s[m.start():m.end()+500].replace('\n',' ')[:900]); print()
