# -*- coding: utf-8 -*-
"""Replace pgfplots figure bodies by \\includegraphics of the python-made figures. usage: swap_figs.py in.tex out.tex label[,label] figdir"""
import re, sys
src, out, labels, figdir = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4]
FILES = {"fig:prev": "fig_prevalence.pdf", "fig:diag": "fig_dsl_diag.pdf"}
s = open(src, encoding="utf-8").read()
for lab in labels:
    m = [f for f in re.finditer(r"\\begin\{figure\}.*?\\end\{figure\}", s, re.S) if f"\\label{{{lab}}}" in f.group(0)]
    assert len(m) == 1, lab
    f = m[0].group(0)
    cap = re.search(r"\\caption\{.*\\label\{" + re.escape(lab) + r"\}", f, re.S).group(0)
    new = "\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{" + figdir + "/" + FILES[lab] + "}\n" + cap + "\n\\end{figure}"
    s = s.replace(f, new)
open(out, "w", encoding="utf-8").write(s); print("ok")
