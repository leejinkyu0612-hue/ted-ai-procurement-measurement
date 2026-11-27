# -*- coding: utf-8 -*-
"""Step 3: M3 zero-shot NLI (MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) entailment probability.
Order: labeled sample first, then the rest of the frame (so M3 evaluation is never blocked).
Checkpointed to data/nli_scores.parquet. Per plan: if full-frame time exceeds 6 h, stop after the sample."""
import os, sys, time, json
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, os.path.dirname(__file__))
from prompts import NLI_HYPOTHESIS
torch.set_num_threads(8)
BASE = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(BASE, "data", "nli_scores.parquet")
name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
tok = AutoTokenizer.from_pretrained(name); mod = AutoModelForSequenceClassification.from_pretrained(name).eval()
ent = [i for i, l in mod.config.id2label.items() if l.lower().startswith("entail")][0]
fr = pd.read_parquet(os.path.join(BASE, "data", "frame.parquet"), columns=["pub", "text", "sampled"])
fr = pd.concat([fr[fr.sampled == 1], fr[fr.sampled == 0]])
done = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=["pub", "m3_score"])
todo = fr[~fr.pub.isin(set(done.pub))]
t0 = time.time(); buf = []; B = 16
with torch.inference_mode():
    for i in range(0, len(todo), B):
        ch = todo.iloc[i:i + B]
        enc = tok(ch.text.tolist(), [NLI_HYPOTHESIS] * len(ch), truncation="only_first", max_length=256, padding=True, return_tensors="pt")
        p = torch.softmax(mod(**enc).logits, -1)[:, ent].numpy()
        buf += list(zip(ch.pub, p.astype(float)))
        if len(buf) >= 160 or i + B >= len(todo):
            done = pd.concat([done, pd.DataFrame(buf, columns=["pub", "m3_score"])]); buf = []
            done.to_parquet(OUT, index=False)
            el = time.time() - t0
            print(len(done), round(el), "s", flush=True)
            if el > 6 * 3600:
                print("time limit reached", flush=True); break
json.dump({"n_scored": len(done), "elapsed_s": time.time() - t0, "threads": 8, "batch": B, "max_length": 256},
          open(os.path.join(BASE, "results", "step3_nli_timing.json"), "w"))

