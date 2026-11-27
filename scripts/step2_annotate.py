# -*- coding: utf-8 -*-
"""Step 2: reference annotation of the stratified sample with two LLMs via OpenRouter.
Every call is logged to logs/annotation_calls.jsonl (request id, model, notice id, raw output, usage).
Resumable: notices already annotated by a model are skipped."""
import os, sys, json, time, re, datetime, urllib.request, concurrent.futures as cf
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from prompts import ANNOTATION_PROMPT

BASE = os.path.join(os.path.dirname(__file__), "..")
LOG = os.path.join(BASE, "logs", "annotation_calls.jsonl")
KEY = open(r"C:\Users\leeji\Desktop\openrouter_apikey.txt", encoding="utf-8").read().strip()
MODELS = {"anthropic/claude-haiku-5.5": {"reasoning": {"enabled": False}},
          "google/gemini-2.5-flash-lite": {}}


def call(model, text):
    body = {"model": model, "temperature": 0, "max_tokens": 30,
            "messages": [{"role": "user", "content": ANNOTATION_PROMPT.format(text=text)}]}
    body.update(MODELS[model])
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def parse(s):
    m = re.search(r'"ai"\s*:\s*([01])', s or "")
    return int(m.group(1)) if m else None


def done_set():
    s = set()
    if os.path.exists(LOG):
        for line in open(LOG, encoding="utf-8"):
            j = json.loads(line)
            if j.get("label") is not None:
                s.add((j["model"], j["pub"]))
    return s


def work(model, pub, text):
    for attempt in range(3):
        try:
            r = call(model, text[:1500])
            out = r["choices"][0]["message"].get("content") or ""
            lab = parse(out)
            rec = {"ts": datetime.datetime.now().isoformat(), "model": model, "served_model": r.get("model"), "id": r.get("id"),
                   "pub": pub, "attempt": attempt + 1, "output": out, "label": lab, "usage": r.get("usage")}
        except Exception as e:
            rec = {"ts": datetime.datetime.now().isoformat(), "model": model, "pub": pub, "attempt": attempt + 1,
                   "error": str(e)[:300], "label": None}
            lab = None
            if "402" in str(e):
                return rec
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if lab is not None:
            return rec
        time.sleep(3)
    return rec


if __name__ == "__main__":
    fp = os.path.join(BASE, "data", "frame.parquet")
    fr = pd.read_parquet(fp if os.path.exists(fp) else os.path.join(BASE, "data", "sample_recon.parquet"), columns=["pub", "text", "sampled"])
    s = fr[fr.sampled == 1]
    done = done_set()
    jobs = [(m, p, t) for m in MODELS for p, t in zip(s.pub, s.text) if (m, p) not in done]
    print("jobs", len(jobs), flush=True)
    with cf.ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(lambda a: work(*a), jobs)):
            if i % 100 == 0:
                print(i, flush=True)
    print("finished", flush=True)

