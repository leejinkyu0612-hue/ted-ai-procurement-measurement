# -*- coding: utf-8 -*-
"""Step 1 (after plan commit 7314e8d): dictionary hits (M1, M2), multilingual-e5-small embeddings,
prototype similarity (M4 score), strata A-D and the reference sample (seed 2027).
Also MiniLM robustness embeddings. Times are logged for the efficiency table."""
import os, sys, time, json
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from dictionaries import EN_RE, ML_RE, hit
from prompts import PROTOTYPES

torch.set_num_threads(8)
BASE = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(BASE, "data"); R = os.path.join(BASE, "results")
df = pd.read_parquet(os.path.join(D, "notices.parquet"))
df = df[(df["title"].str.len() + df["desc"].str.len()) >= 10].reset_index(drop=True)
timing = {}
t = time.time(); df["m1"] = [hit(x, EN_RE) for x in df["text"]]; timing["M1_s"] = time.time() - t
t = time.time(); df["m2"] = [hit(x, ML_RE) for x in df["text"]]; timing["M2_s"] = time.time() - t
print("frame", len(df), "m1", df.m1.sum(), "m2", df.m2.sum(), flush=True)

from sentence_transformers import SentenceTransformer
def embed(name, texts, prefix=""):
    m = SentenceTransformer(name, device="cpu")
    m.max_seq_length = 256
    t = time.time()
    e = m.encode([prefix + x for x in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return m, e.astype(np.float32), time.time() - t

m, E, tt = embed("intfloat/multilingual-e5-small", df["text"].tolist(), "passage: ")
timing["e5_small_embed_s"] = tt
np.save(os.path.join(D, "emb_e5small.npy"), E)
P = m.encode(PROTOTYPES, normalize_embeddings=True, convert_to_numpy=True)
df["m4_score"] = (E @ P.T).max(axis=1)
print("e5 done", tt, flush=True)

# strata
thr = np.quantile(df["m4_score"], 0.97)
df["stratum"] = np.where(df.m1 == 1, "A", np.where(df.m2 == 1, "B", np.where(df.m4_score >= thr, "C", "D")))
rng = np.random.default_rng(2027)
target = {"A": 250, "B": 250, "C": 250, "D": 450}
df["sampled"] = 0; df["pi"] = np.nan
for h, n in target.items():
    idx = df.index[df.stratum == h].to_numpy()
    k = min(n, len(idx))
    pick = rng.choice(idx, size=k, replace=False)
    df.loc[pick, "sampled"] = 1
    df.loc[df.stratum == h, "pi"] = k / len(idx)
strat = df.groupby("stratum").agg(N=("pub", "size"), n=("sampled", "sum"), pi=("pi", "first")).reset_index().to_dict("records")
print(strat, flush=True)

m2_, E2, tt2 = embed("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", df["text"].tolist())
timing["minilm_embed_s"] = tt2
np.save(os.path.join(D, "emb_minilm.npy"), E2)
P2 = m2_.encode([p.replace("query: ", "") for p in PROTOTYPES], normalize_embeddings=True, convert_to_numpy=True)
df["m4_score_minilm"] = (E2 @ P2.T).max(axis=1)
df.to_parquet(os.path.join(D, "frame.parquet"), index=False)
json.dump({"frame_n": len(df), "m1_hits": int(df.m1.sum()), "m2_hits": int(df.m2.sum()), "m4_top3pct_threshold": float(thr),
           "strata": strat, "timing_s": timing, "n_cpu_threads": 8}, open(os.path.join(R, "step1_sample.json"), "w"), indent=1)
print("done", timing, flush=True)
