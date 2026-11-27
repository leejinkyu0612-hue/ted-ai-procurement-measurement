# -*- coding: utf-8 -*-
"""Chapter split / style-check / reassemble for LaTeX manuscripts (S12, S16, S23).
split:  python style_tools.py split <in.tex> <outdir>
check:  python style_tools.py merge <in.tex> <outdir> <out.tex> <report.md>
outdir/chap_XX_name.txt  : chapter text sent to ChatGPT (environments protected as [[KEEP_n]])
outdir/chap_XX_name.resp.txt : ChatGPT output (rewritten chapter only)
Rules (per paragraph): numbers, \\cite/\\ref keys, KEEP tokens, and math must be identical, and the word count must not
increase; otherwise the original paragraph is kept."""
import re, sys, os, json

PROTECT = re.compile(r"(\\begin\{(table|figure|equation|tabular)\*?\}.*?\\end\{\2\*?\}|\\label\{[^}]*\})", re.S)


def chapters(tex):
    body = tex
    out = []
    m = re.search(r"\\begin\{abstract\}(.*?)\\keywords", body, re.S)
    out.append(("00_Abstract", m.start(1), m.end(1)))
    secs = list(re.finditer(r"\\section\{([^}]*)\}", body))
    endb = body.find("\\begin{credits}")
    if endb < 0: endb = body.find("\\bibliographystyle")
    for i, s in enumerate(secs):
        st = s.end(); en = secs[i + 1].start() if i + 1 < len(secs) else endb
        name = re.sub(r"[^A-Za-z]+", "_", s.group(1)).strip("_")
        out.append((f"{i+1:02d}_{name}", st, en))
    return out


def protect(txt):
    keeps = []
    def rep(m):
        keeps.append(m.group(0)); return f"[[KEEP_{len(keeps)-1}]]"
    return PROTECT.sub(rep, txt), keeps


def nums(s):
    s = re.sub(r"\\(cite|ref|label|eqref)\{[^}]*\}", "", s)
    return sorted(re.findall(r"\d+(?:[.,]\d+)*", s))


def keys(s):
    return sorted(re.findall(r"\\(?:cite|ref|eqref)\{([^}]*)\}", s))


def maths(s):
    return sorted(re.findall(r"\$[^$]*\$", s))


def words(s):
    return len(re.findall(r"[A-Za-z0-9]+", re.sub(r"\\[a-zA-Z]+", "", s)))


def paras(s):
    return [p for p in re.split(r"\n\s*\n", s.strip())]


if __name__ == "__main__":
    cmd = sys.argv[1]
    tex = open(sys.argv[2], encoding="utf-8").read()
    od = sys.argv[3]; os.makedirs(od, exist_ok=True)
    chs = chapters(tex)
    if cmd == "split":
        meta = []
        for name, a, b in chs:
            p, keeps = protect(tex[a:b])
            open(os.path.join(od, f"chap_{name}.txt"), "w", encoding="utf-8").write(p.strip())
            json.dump(keeps, open(os.path.join(od, f"chap_{name}.keeps.json"), "w", encoding="utf-8"), ensure_ascii=False)
            meta.append(name)
        print(meta)
    elif cmd == "merge":
        outp, rep = sys.argv[4], sys.argv[5]
        new = tex; lines = ["| Chapter | paragraphs | adopted | reverted | reasons | words before | words after |", "|---|---|---|---|---|---|---|"]
        tot_b = tot_a = 0
        for name, a, b in reversed(chs):
            orig_p, keeps = protect(tex[a:b])
            rp = os.path.join(od, f"chap_{name}.resp.txt")
            if not os.path.exists(rp):
                continue
            resp = open(rp, encoding="utf-8").read().strip()
            resp = re.sub(r"^```[a-z]*\n|\n```$", "", resp)
            op, npars = paras(orig_p), paras(resp)
            adopted = reverted = 0; reasons = []; final = []
            if len(op) != len(npars):
                final = op; reverted = len(op); reasons.append(f"paragraph count {len(op)}->{len(npars)}")
            else:
                for o, n in zip(op, npars):
                    why = []
                    if nums(o) != nums(n): why.append("numbers")
                    if keys(o) != keys(n): why.append("cite/ref")
                    if sorted(re.findall(r"\[\[KEEP_\d+\]\]", o)) != sorted(re.findall(r"\[\[KEEP_\d+\]\]", n)): why.append("keep")
                    if maths(o) != maths(n): why.append("math")
                    if words(n) > words(o): why.append("longer")
                    if why: final.append(o); reverted += 1; reasons += why
                    else: final.append(n); adopted += 1
            txt = "\n\n".join(final)
            for i, k in enumerate(keeps):
                txt = txt.replace(f"[[KEEP_{i}]]", k)
            wb, wa = words(orig_p), words("\n\n".join(final))
            tot_b += wb; tot_a += wa
            new = new[:a] + "\n" + txt + "\n\n" + new[b:]
            from collections import Counter
            lines.insert(2, f"| {name} | {len(op)} | {adopted} | {reverted} | {dict(Counter(reasons))} | {wb} | {wa} |")
        open(outp, "w", encoding="utf-8").write(new)
        lines.append(f"\nTotal words in edited chapters: {tot_b} -> {tot_a}")
        lines.append(f"Whole-manuscript number multiset identical: {nums(tex) == nums(new)}; citation keys identical: {keys(tex) == keys(new)}")
        open(rep, "w", encoding="utf-8").write("\n".join(lines))
        print("\n".join(lines))
