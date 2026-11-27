# reconstruct the step1 reference sample (identical code path) so annotation can start before MiniLM finishes
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from dictionaries import EN_RE, ML_RE, hit
from prompts import PROTOTYPES
from sentence_transformers import SentenceTransformer
df = pd.read_parquet("data/notices.parquet")
df = df[(df["title"].str.len() + df["desc"].str.len()) >= 10].reset_index(drop=True)
df["m1"] = [hit(x, EN_RE) for x in df["text"]]; df["m2"] = [hit(x, ML_RE) for x in df["text"]]
E = np.load("data/emb_e5small.npy")
m = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
P = m.encode(PROTOTYPES, normalize_embeddings=True, convert_to_numpy=True)
df["m4_score"] = (E @ P.T).max(axis=1)
thr = np.quantile(df["m4_score"], 0.97)
df["stratum"] = np.where(df.m1 == 1, "A", np.where(df.m2 == 1, "B", np.where(df.m4_score >= thr, "C", "D")))
rng = np.random.default_rng(2027); df["sampled"] = 0
for h, n in {"A": 250, "B": 250, "C": 250, "D": 450}.items():
    idx = df.index[df.stratum == h].to_numpy(); k = min(n, len(idx))
    df.loc[rng.choice(idx, size=k, replace=False), "sampled"] = 1
df[["pub", "text", "sampled", "stratum"]].to_parquet("data/sample_recon.parquet", index=False)
print(df.groupby("stratum").sampled.agg(["size", "sum"]))
