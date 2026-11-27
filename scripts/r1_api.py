# -*- coding: utf-8 -*-
"""R1 (added after ChatGPT review R1; POST HOC).
(a) Re-query the secondary annotator (gemini-2.5-flash-lite) for notices whose secondary output was unparsable,
    with max_tokens = 120 (otherwise identical prompt), to recover labels. Logged separately.
(b) Third-model adjudication (deepseek/deepseek-chat-v3.1, temperature 0, same prompt) for every labeled notice where
    primary and secondary disagree or the secondary is missing, and for all primary positives in strata C and D.
All calls -> logs/r1_calls.jsonl"""
import os, sys, json, re, time, datetime, urllib.request, concurrent.futures as cf
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prompts import ANNOTATION_PROMPT
BASE = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(BASE, "logs", "r1_calls.jsonl")
KEY = open(r"C:\Users\leeji\Desktop\openrouter_apikey.txt", encoding="utf-8").read().strip()


def call(model, text, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": ANNOTATION_PROMPT.format(text=text[:1500])}]}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def parse(s):
    m = re.search(r'"ai"\s*:\s*([01])', s or ""); return int(m.group(1)) if m else None


def work(task, model, pub, text, mt):
    for a in range(3):
        try:
            r = call(model, text, mt); out = r["choices"][0]["message"].get("content") or ""; lab = parse(out)
            rec = {"ts": datetime.datetime.now().isoformat(), "task": task, "model": model, "served_model": r.get("model"), "id": r.get("id"),
                   "pub": pub, "attempt": a + 1, "max_tokens": mt, "output": out, "label": lab, "usage": r.get("usage")}
        except Exception as e:
            rec = {"ts": datetime.datetime.now().isoformat(), "task": task, "model": model, "pub": pub, "attempt": a + 1, "error": str(e)[:300], "label": None}; lab = None
        with open(LOG, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if lab is not None: return rec
        time.sleep(3)


lab = {}
for line in open(os.path.join(BASE, "logs", "annotation_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is not None: lab[(j["model"], j["pub"])] = j["label"]
fr = pd.read_parquet(os.path.join(BASE, "data", "frame.parquet"), columns=["pub", "text", "sampled", "stratum"])
s = fr[fr.sampled == 1].copy()
s["yp"] = [lab.get(("anthropic/claude-haiku-5.5", p)) for p in s.pub]; s["ys"] = [lab.get(("google/gemini-2.5-flash-lite", p)) for p in s.pub]
jobs = [("reparse_secondary", "google/gemini-2.5-flash-lite", r.pub, r.text, 120) for r in s.itertuples() if pd.isna(r.ys)]
adj = s[(s.yp.notna()) & ((s.ys.isna()) | (s.yp != s.ys) | ((s.stratum.isin(["C", "D"])) & (s.yp == 1)))]
jobs_adj = [("adjudicate", "deepseek/deepseek-chat-v3.1", r.pub, r.text, 30) for r in adj.itertuples()]
import sys as _s
if '--reparse-only' not in _s.argv: jobs += jobs_adj
print('jobs', len(jobs), flush=True)
with cf.ThreadPoolExecutor(4) as ex: list(ex.map(lambda a: work(*a), jobs))
print("done")


