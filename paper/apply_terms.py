# -*- coding: utf-8 -*-
"""S13: apply Claude's term substitutions (ordered) with author overrides; log every application; build the xlsx."""
import json, re, sys
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
src = open("S12.tex", encoding="utf-8").read()
subs = json.load(open("s13_subs.json", encoding="utf-8"))
SKIP = {"For the linear probability model", "A linear probability model is used because"}
OVERRIDE = {"Coefficient on AI-relatedness in a linear probability model": "Coefficient on the AI indicator in a linear probability model"}
EXTRA = [("operationalised", "operationalized"), ("partialling", "partialing"), ("modelled", "modeled"), ("summarised", "summarized"),
         ("normalisation", "normalization"), ("recognised", "recognized"), ("prioritised", "prioritized"), ("behaviour", "behavior"),
         ("organisation", "organization"), ("characterised", "characterized"), ("minimise", "minimize"), ("emphasise", "emphasize"),
         ("parametrised", "parametrized"), ("randomised", "randomized"), ("stabilise", "stabilize"), ("utilise", "utilize"), ("centred", "centered")]
body_start = src.find("\\begin{document}")
log = []
s = src
for it in subs:
    old, new = it["old"], OVERRIDE.get(it["old"], it["new"])
    if it["old"] in SKIP:
        log.append((old, new, 0, "skipped (author: linear probability model is the standard econometric term)")); continue
    n = s.count(old)
    if n == 0:
        log.append((old, new, 0, "not found")); continue
    s = s.replace(old, new); log.append((old, new, n, "applied"))
for old, new in EXTRA:
    n = len(re.findall(old, s))
    if n:
        s = re.sub(old, new, s); log.append((old, new, n, "applied (spelling consistency)"))
# guard: numbers/citations unchanged
def nums(t):
    t = re.sub(r"\\(cite|ref|label|eqref|orcidID|url)\{[^}]*\}", "", t); return sorted(re.findall(r"\d+(?:[.,]\d+)*", t))
def keys(t): return sorted(re.findall(r"\\(?:cite|ref|eqref)\{([^}]*)\}", t))
ok_n, ok_k = nums(src) == nums(s), keys(src) == keys(s)
open("S13.tex", "w", encoding="utf-8").write(s)
# xlsx
tbl = [l for l in open("s13_table.md", encoding="utf-8").read().split("\n") if re.match(r"^\| *\d+ *\|", l)]
wb = Workbook(); ws = wb.active; ws.title = "용어대조표"
hdr = ["#", "A 논문에 쓰인 용어", "B 선행연구 용어(등장 논문 수 / 22편)", "C 바꾼 용어", "판정·이유"]
ws.append(hdr)
for c in ws[1]: c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="2F5D8A"); c.alignment = Alignment(wrap_text=True, vertical="center")
for l in tbl:
    cells = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(cells) < 5: continue
    num, a, b, cc, why = cells[:5]
    if "linear probability model" in a and "OLS" in cc:
        cc, why = "유지(linear probability model)", why + " / 저자 판단: 계량경제학 표준 용어라 유지(치환 미적용)"
    ws.append([int(num), a, b, cc, why])
    fill = "FCE4D6" if not cc.startswith("유지") else None
    if fill:
        for c in ws[ws.max_row]: c.fill = PatternFill("solid", fgColor=fill)
for col, w in zip("ABCDE", [5, 38, 42, 38, 60]): ws.column_dimensions[col].width = w
for row in ws.iter_rows(min_row=2):
    for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
ws2 = wb.create_sheet("치환 적용 기록"); ws2.append(["old", "new", "적용 횟수", "상태"])
for r in log: ws2.append(list(r))
ws2.append([]); ws2.append(["숫자 목록 동일", str(ok_n)]); ws2.append(["인용 키 동일", str(ok_k)])
ws3 = wb.create_sheet("선행연구 목록(22편)")
for p in json.load(open("../../S13_용어변경/선행연구_용어빈도.json", encoding="utf-8"))["papers"]: ws3.append([p])
wb.save("../../S13_용어변경/용어대조표.xlsx")
print("applied", sum(1 for r in log if r[3].startswith("applied")), "not found", [r[0][:50] for r in log if r[3] == "not found"], "numbers same", ok_n, "keys same", ok_k)
