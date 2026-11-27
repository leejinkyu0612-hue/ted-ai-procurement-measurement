# -*- coding: utf-8 -*-
"""R3 analyses (added after ChatGPT review R3; POST HOC). Primary reference X, design pi.
1 full-frame cross-fitting: every frame notice assigned to one of 5 folds (stratified by stratum); g for a notice comes
  from the model trained on labelled notices outside its fold. DSL prevalence and DSL-LPM recomputed.
2 Schur complement of A for the AI column (effective information after partialling out controls), full vs reduced controls.
3 bootstrap with refitting of M5 (and threshold-free g) inside each replicate for DSL prevalence (500 replicates).
4 bounds for the unparsed stratum-C notice (coded 0 or 1).
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
BASE = os.path.join(os.path.dirname(__file__), ".."); D = os.path.join(BASE, "data"); R = os.path.join(BASE, "results")
SEED = 2027
fr = pd.read_parquet(os.path.join(D, "frame.parquet")); N = len(fr); E = np.load(os.path.join(D, "emb_e5small.npy"))
lab = {}
for line in open(os.path.join(BASE, "logs", "annotation_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is not None: lab[(j["model"], j["pub"])] = j["label"]
fr["y"] = [lab.get(("anthropic/claude-haiku-5.5", p)) for p in fr.pub]
S = fr.index[(fr.sampled == 1) & fr.y.notna()].to_numpy()
Nh = fr.groupby("stratum").size(); nh = fr[fr.sampled == 1].groupby("stratum").size()
fr["pi"] = fr.stratum.map((nh / Nh).to_dict())
w = 1 / fr.loc[S, "pi"].to_numpy(); st = fr.loc[S, "stratum"].to_numpy(); y = fr.y.to_numpy(dtype=float); yS = y[S].astype(int)
out = {"note": "R3 analyses, post hoc"}
# 1 full-frame cross-fitting
fold_all = np.zeros(N, int)
for h in Nh.index:
    idx = fr.index[fr.stratum == h].to_numpy(); rng = np.random.default_rng(SEED); perm = rng.permutation(idx)
    fold_all[perm] = np.arange(len(perm)) % 5
g_full = np.zeros(N)
isL = np.zeros(N, bool); isL[S] = True
for k in range(5):
    tr = S[fold_all[S] != k]
    m = LogisticRegression(C=1.0, max_iter=3000).fit(E[tr], y[tr].astype(int), sample_weight=1 / fr.loc[tr, "pi"].to_numpy())
    te = np.where(fold_all == k)[0]; g_full[te] = m.predict_proba(E[te])[:, 1]
dsl = (g_full.sum() + (w * (yS - g_full[S])).sum()) / N
rngb = np.random.default_rng(SEED)
boot = [np.concatenate([rngb.choice(np.where(st == h)[0], (st == h).sum(), replace=True) for h in np.unique(st)]) for _ in range(2000)]
dbs = [(g_full.sum() + (w[b] * (yS[b] - g_full[S][b])).sum()) / N for b in boot]
out["fullframe_crossfit_prevalence"] = {"dsl": float(dsl), "ci": [float(np.percentile(dbs, 2.5)), float(np.percentile(dbs, 97.5))], "H3_diff": float(dsl - fr.m1.mean())}
RS = fr.index[(fr.n_tender_fields == 1) & (fr.tenders >= 1) & (fr.tenders <= 100)].to_numpy()
Y = (fr.loc[RS, "tenders"].to_numpy() == 1).astype(float)
cpv = fr.loc[RS, "cpv_div"].fillna("NA"); vc = cpv.value_counts(); cpv = cpv.where(cpv.map(vc) >= 30, "other")
blocks = {"country": pd.get_dummies(fr.loc[RS, "country"].fillna("NA"), prefix="c", drop_first=True), "cpv": pd.get_dummies(cpv, prefix="cpv", drop_first=True),
          "proc": pd.get_dummies(fr.loc[RS, "procedure"].fillna("NA"), prefix="p", drop_first=True), "buyer": pd.get_dummies(fr.loc[RS, "buyer_type"].fillna("NA"), prefix="b", drop_first=True),
          "q": pd.get_dummies(fr.loc[RS, "quarter"], prefix="q", drop_first=True)}
val = pd.to_numeric(fr.loc[RS, "value"], errors="coerce")
vb = pd.DataFrame({"logv": np.log1p(val.fillna(0).clip(lower=0)), "vmiss": val.isna().astype(float)}, index=RS)
def design(keys):
    parts = [blocks[k] for k in keys if k != "value"] + ([vb] if "value" in keys else [])
    Wd = pd.concat(parts, axis=1).astype(float) if parts else pd.DataFrame(index=RS); Wd["const"] = 1.0; return Wd.to_numpy()
lab_rs = isL[RS]; pi_rs = fr.loc[RS, "pi"].to_numpy(); x_rs = np.nan_to_num(y[RS])
def dsl_lpm(g, Wm):
    gR = g[RS]; xt = gR.copy(); xt[lab_rs] = gR[lab_rs] + (x_rs[lab_rs] - gR[lab_rs]) / pi_rs[lab_rs]
    Z = np.column_stack([xt, Wm]); A = Z.T @ Z; A[0, 0] = xt.sum(); beta = np.linalg.lstsq(A, Z.T @ Y, rcond=None)[0]
    psi = Z * (Y - Z @ beta)[:, None]; psi[:, 0] = xt * Y - xt * beta[0] - xt * (Wm @ beta[1:])
    Ai = np.linalg.pinv(A); se = float(np.sqrt((Ai @ (psi.T @ psi) @ Ai)[0, 0]))
    Axx = A[0, 0]; AxW = A[0, 1:]; AWW = Wm.T @ Wm
    schur = float(Axx - AxW @ np.linalg.pinv(AWW) @ AxW)
    return {"beta_pp": float(beta[0] * 100), "se_pp": se * 100, "schur": schur, "Axx": float(Axx)}
full = ["country", "cpv", "proc", "buyer", "q", "value"]
out["fullframe_crossfit_lpm_full"] = dsl_lpm(g_full, design(full))
# 2 Schur complements with the pre-specified g (labelled cross-fitted, unlabelled full model)
g_pre = LogisticRegression(C=1.0, max_iter=3000).fit(E[S], yS, sample_weight=w).predict_proba(E)[:, 1]
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(S, st):
    g_pre[S[te]] = LogisticRegression(C=1.0, max_iter=3000).fit(E[S[tr]], yS[tr], sample_weight=w[tr]).predict_proba(E[S[te]])[:, 1]
out["schur"] = {name: dsl_lpm(g_pre, design(keys)) for name, keys in [("no_controls", []), ("country_only", ["country"]), ("full", full)]}
# 3 refit bootstrap for DSL prevalence
rb = []
rngr = np.random.default_rng(SEED + 1)
for r in range(500):
    b = np.concatenate([rngr.choice(np.where(st == h)[0], (st == h).sum(), replace=True) for h in np.unique(st)])
    Sb, yb, wb = S[b], yS[b], w[b]
    if yb.sum() < 5: continue
    gb = LogisticRegression(C=1.0, max_iter=3000).fit(E[Sb], yb, sample_weight=wb).predict_proba(E)[:, 1]
    rb.append((gb.sum() + (wb * (yb - gb[Sb])).sum()) / N)
out["refit_bootstrap_dsl_prevalence"] = {"reps": len(rb), "ci": [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))], "se": float(np.std(rb))}
# 4 bounds for unparsed stratum-C notice
miss = fr.index[(fr.sampled == 1) & fr.y.isna()].to_numpy()
bounds = {}
for v in (0, 1):
    S2 = np.concatenate([S, miss]); y2 = np.concatenate([yS, np.full(len(miss), v)]); w2 = 1 / fr.loc[S2, "pi"].to_numpy()
    bounds[f"coded_{v}"] = {"HT": float((w2 * y2).sum() / N)}
out["unparsed_C_bounds"] = bounds
out["HT_reference"] = float((w * yS).sum() / N)
json.dump(out, open(os.path.join(R, "r3_analyses.json"), "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
