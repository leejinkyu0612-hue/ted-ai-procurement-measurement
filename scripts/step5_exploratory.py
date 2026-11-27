# -*- coding: utf-8 -*-
"""Step 5 (POST HOC / EXPLORATORY). Added after the confirmatory run showed that the pre-specified
design-weighted M5/M6 classifiers collapsed to very low recall at threshold 0.5.
Not part of the confirmatory tests.
(a) M5u/M6u: same features and folds, logistic regression trained WITHOUT design weights; thresholds 0.5 and
    a fold-internal F1-optimal threshold; evaluated with design weights (as all methods).
(b) DSL prevalence and DSL-LPM using M5u probabilities as g (DSL validity does not depend on how g is trained).
(c) Stratum decomposition of estimated positives.
(d) M3 metrics on the labeled sample (NLI scores computed for labeled notices only; Amendment 1).
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

BASE = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(BASE, "data"); R = os.path.join(BASE, "results")
SEED = 2027; NB = 2000
fr = pd.read_parquet(os.path.join(D, "frame.parquet")); N = len(fr)
E = np.load(os.path.join(D, "emb_e5small.npy"))
lab = {}
for line in open(os.path.join(BASE, "logs", "annotation_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is not None: lab[(j["model"], j["pub"])] = j["label"]
fr["y_p"] = [lab.get(("anthropic/claude-haiku-5.5", p)) for p in fr.pub]
nli_p = os.path.join(D, "nli_scores.parquet")
fr = fr.merge(pd.read_parquet(nli_p), on="pub", how="left") if os.path.exists(nli_p) else fr.assign(m3_score=np.nan)
S = fr.index[(fr.sampled == 1) & fr.y_p.notna()].to_numpy()
Nh = fr.groupby("stratum").size(); nh = fr[fr.sampled == 1].groupby("stratum").size()
fr["pi_d"] = fr.stratum.map((nh / Nh).to_dict())
w = 1 / fr.loc[S, "pi_d"].to_numpy(); st = fr.loc[S, "stratum"].to_numpy(); y = fr.y_p.to_numpy(dtype=float); yS = y[S].astype(int)
eng = (fr.lang == "ENG").to_numpy()
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(S, st))
rng = np.random.default_rng(SEED)
boot = [np.concatenate([rng.choice(np.where(st == h)[0], (st == h).sum(), replace=True) for h in np.unique(st)]) for _ in range(NB)]


def met(p, yy, ww, m=None):
    m = np.ones(len(yy), bool) if m is None else m
    p, yy, ww = p[m] == 1, yy[m] == 1, ww[m]
    tp = (ww * (p & yy)).sum(); fp = (ww * (p & ~yy)).sum(); fn = (ww * (~p & yy)).sum()
    return (tp / (tp + fp) if tp + fp else np.nan, tp / (tp + fn) if tp + fn else np.nan, 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def best_thr(sc, yy, ww):
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(sc, np.linspace(0.3, 0.999, 300))):
        f = met((sc >= t).astype(int), yy.astype(float), ww)[2]
        if f > best: best, bt = f, t
    return bt


def crossfit(X):
    full = LogisticRegression(C=1.0, max_iter=3000).fit(X[S], yS)
    g = full.predict_proba(X)[:, 1]; thr_lab = np.zeros(len(S))
    for tr, te in folds:
        m = LogisticRegression(C=1.0, max_iter=3000).fit(X[S[tr]], yS[tr])
        g[S[te]] = m.predict_proba(X[S[te]])[:, 1]
        thr_lab[te] = best_thr(m.predict_proba(X[S[tr]])[:, 1], yS[tr], w[tr])
    return g, thr_lab, best_thr(g[S], yS, w)


out = {"note": "POST HOC / EXPLORATORY"}
g5, thr5, thr5_all = crossfit(E)
T = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True).fit(fr.text.iloc[S]).transform(fr.text).tocsr()
g6, thr6, _ = crossfit(T)
cands = {"M5u@0.5": (g5[S] >= 0.5).astype(int), "M5u@cv": (g5[S] >= thr5).astype(int),
         "M6u@0.5": (g6[S] >= 0.5).astype(int), "M6u@cv": (g6[S] >= thr6).astype(int),
         "M1_en_dict": fr.m1.to_numpy()[S], "M2_ml_dict": fr.m2.to_numpy()[S]}
if fr.loc[S, "m3_score"].notna().all():
    sc3 = fr.loc[S, "m3_score"].to_numpy()
    for t in (0.3, 0.5, 0.7): cands[f"M3_nli@{t}"] = (sc3 >= t).astype(int)
    out["auprc_M3"] = float(average_precision_score(yS, sc3, sample_weight=w))
res = {}
for k, p in cands.items():
    pr, rc, f1 = met(p, yS.astype(float), w); bs = [met(p[b], yS[b].astype(float), w[b]) for b in boot]
    res[k] = {"precision": pr, "recall": rc, "f1": f1, "precision_ci": ci([b[0] for b in bs]), "recall_ci": ci([b[1] for b in bs]), "f1_ci": ci([b[2] for b in bs]),
              "recall_eng": met(p, yS.astype(float), w, eng[S])[1], "recall_noneng": met(p, yS.astype(float), w, ~eng[S])[1]}
out["metrics"] = res
out["auprc"] = {"M5u": float(average_precision_score(yS, g5[S], sample_weight=w)), "M6u": float(average_precision_score(yS, g6[S], sample_weight=w))}
for base in ["M1_en_dict", "M2_ml_dict"]:
    d = [met(cands["M5u@cv"][b], yS[b].astype(float), w[b])[2] - met(cands[base][b], yS[b].astype(float), w[b])[2] for b in boot]
    out[f"diff_f1_M5ucv_minus_{base}"] = {"est": res["M5u@cv"]["f1"] - res[base]["f1"], "ci": ci(d)}
corr = w * (yS - g5[S]); dsl = (g5.sum() + corr.sum()) / N
out["prevalence_dsl_g_M5u"] = {"est": float(dsl), "ci": ci([(g5.sum() + (w[b] * (yS[b] - g5[S][b])).sum()) / N for b in boot]),
                              "naive_M5u@cv_full_frame": float((g5 >= thr5_all).mean())}
RS = fr.index[(fr.n_tender_fields == 1) & (fr.tenders >= 1) & (fr.tenders <= 100)].to_numpy()
Y = (fr.loc[RS, "tenders"].to_numpy() == 1).astype(float)
cpv = fr.loc[RS, "cpv_div"].fillna("NA"); vc = cpv.value_counts(); cpv = cpv.where(cpv.map(vc) >= 30, "other")
Wd = pd.concat([pd.get_dummies(fr.loc[RS, "country"].fillna("NA"), prefix="c", drop_first=True), pd.get_dummies(cpv, prefix="cpv", drop_first=True),
                pd.get_dummies(fr.loc[RS, "procedure"].fillna("NA"), prefix="p", drop_first=True), pd.get_dummies(fr.loc[RS, "buyer_type"].fillna("NA"), prefix="b", drop_first=True),
                pd.get_dummies(fr.loc[RS, "quarter"], prefix="q", drop_first=True)], axis=1).astype(float)
val = pd.to_numeric(fr.loc[RS, "value"], errors="coerce"); Wd["logv"] = np.log1p(val.fillna(0).clip(lower=0)); Wd["vmiss"] = val.isna().astype(float); Wd["const"] = 1.0
Wm = Wd.to_numpy()
pos = {s: k for k, s in enumerate(S)}
xt = g5[RS].copy(); nlab = 0
for k, i in enumerate(RS):
    if i in pos: xt[k] = g5[i] + (y[i] - g5[i]) / fr.at[i, "pi_d"]; nlab += 1
Z = np.column_stack([xt, Wm]); A = Z.T @ Z; A[0, 0] = xt.sum(); beta = np.linalg.solve(A + 1e-9 * np.eye(len(A)), Z.T @ Y)
psi = Z * (Y - Z @ beta)[:, None]; psi[:, 0] = xt * Y - xt * beta[0] - xt * (Wm @ beta[1:])
Ai = np.linalg.pinv(A); se = float(np.sqrt((Ai @ (psi.T @ psi) @ Ai)[0, 0]))
out["H4_DSL_g_M5u"] = {"beta": float(beta[0]), "se": se, "ci": [float(beta[0] - 1.96 * se), float(beta[0] + 1.96 * se)], "n_labeled_in_reg": nlab, "n": int(len(RS))}
xb = (g5[RS] >= thr5_all).astype(float); Z2 = np.column_stack([xb, Wm]); A2 = Z2.T @ Z2; b2 = np.linalg.lstsq(A2, Z2.T @ Y, rcond=None)[0]
e2 = Y - Z2 @ b2; P2 = Z2 * e2[:, None]; A2i = np.linalg.pinv(A2); se2 = float(np.sqrt((A2i @ (P2.T @ P2) @ A2i)[0, 0]))
out["H4_naive_M5u@cv"] = {"beta": float(b2[0]), "se": se2, "ci": [float(b2[0] - 1.96 * se2), float(b2[0] + 1.96 * se2)], "n_ai": int(xb.sum())}
dec = {}
for h in ["A", "B", "C", "D"]:
    m = st == h; dec[h] = {"n_labeled": int(m.sum()), "n_pos": int(yS[m].sum()), "est_pos_in_frame": float((w[m] * yS[m]).sum())}
out["stratum_decomposition"] = dec
out["thresholds"] = {"M5u_cv_full": float(thr5_all)}
json.dump(out, open(os.path.join(R, "exploratory_posthoc.json"), "w"), indent=1, default=float)
for k, v in res.items(): print(k, v["precision"], v["recall"], v["f1"], v["f1_ci"])
print(json.dumps({k: out[k] for k in out if k != "metrics"}, indent=1, default=float))
