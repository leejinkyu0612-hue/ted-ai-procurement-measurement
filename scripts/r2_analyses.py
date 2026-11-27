# -*- coding: utf-8 -*-
"""R2 analyses (added after ChatGPT review R2; POST HOC). Primary reference X unless noted.
1 HT-based H3 (HT minus M1 share) with stratified bootstrap CI
2 exact (Clopper-Pearson) CI for the stratum-D positive rate and its implied contribution
3 2x2 weighting x threshold for M5 (completes the ablation)
4 DSL-LPM instability diagnosis: fixed-effect sets (none / country / full) and auxiliary predictors (M5, M5u, M2 indicator)
5 single-bidding construct: share by procedure type; naive and labelled-only estimates restricted to open procedures
6 selection: M1/M2 shares and stratum composition in the regression sample vs excluded frame notices
"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import beta as Beta
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
BASE = os.path.join(os.path.dirname(__file__), ".."); D = os.path.join(BASE, "data"); R = os.path.join(BASE, "results")
SEED = 2027; NB = 2000
fr = pd.read_parquet(os.path.join(D, "frame.parquet")); N = len(fr); E = np.load(os.path.join(D, "emb_e5small.npy"))
lab = {}
for line in open(os.path.join(BASE, "logs", "annotation_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is not None: lab[(j["model"], j["pub"])] = j["label"]
fr["y"] = [lab.get(("anthropic/claude-haiku-5.5", p)) for p in fr.pub]
S = fr.index[(fr.sampled == 1) & fr.y.notna()].to_numpy()
Nh = fr.groupby("stratum").size(); nh = fr[fr.sampled == 1].groupby("stratum").size()  # design (drawn) counts, as pre-specified
fr["pi"] = fr.stratum.map((nh / Nh).to_dict())
w = 1 / fr.loc[S, "pi"].to_numpy(); st = fr.loc[S, "stratum"].to_numpy(); y = fr.y.to_numpy(dtype=float); yS = y[S].astype(int)
rng = np.random.default_rng(SEED)
boot = [np.concatenate([rng.choice(np.where(st == h)[0], (st == h).sum(), replace=True) for h in np.unique(st)]) for _ in range(NB)]
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
out = {"note": "R2 analyses, post hoc"}
m1s = fr.m1.mean()
ht = (w * yS).sum() / N
out["H3_HT"] = {"HT": float(ht), "M1_share": float(m1s), "diff": float(ht - m1s), "ci": ci([(w[b] * yS[b]).sum() / N - m1s for b in boot])}
dpos = int(yS[st == "D"].sum()); dn = int((st == "D").sum())
lo, hi = Beta.ppf(0.025, dpos, dn - dpos + 1), Beta.ppf(0.975, dpos + 1, dn - dpos)
out["stratum_D_exact"] = {"pos": dpos, "n": dn, "rate": dpos / dn, "cp_ci": [float(lo), float(hi)], "implied_frame_positives": [float(lo * Nh["D"]), float(dpos / dn * Nh["D"]), float(hi * Nh["D"])],
                          "implied_frame_share_contribution_pct": [float(lo * Nh["D"] / N * 100), float(hi * Nh["D"] / N * 100)]}
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(S, st))


def met(p, yy, ww):
    p, yy = p == 1, yy == 1; tp = (ww * (p & yy)).sum(); fp = (ww * (p & ~yy)).sum(); fn = (ww * (~p & yy)).sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan


def best_thr(sc, yy, ww):
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(sc, np.linspace(0.3, 0.999, 300))):
        f = met((sc >= t).astype(int), yy, ww)
        if f > best: best, bt = f, t
    return bt


def cross(weighted):
    g = LogisticRegression(C=1.0, max_iter=3000).fit(E[S], yS, sample_weight=w if weighted else None).predict_proba(E)[:, 1]; thr = np.zeros(len(S))
    for tr, te in folds:
        m = LogisticRegression(C=1.0, max_iter=3000).fit(E[S[tr]], yS[tr], sample_weight=w[tr] if weighted else None)
        g[S[te]] = m.predict_proba(E[S[te]])[:, 1]; thr[te] = best_thr(m.predict_proba(E[S[tr]])[:, 1], yS[tr], w[tr])
    return g, thr


gw, tw = cross(True); gu, tu = cross(False)
out["ablation_2x2_F1"] = {"weighted@0.5": met((gw[S] >= 0.5).astype(int), yS, w), "weighted@cv": met((gw[S] >= tw).astype(int), yS, w),
                          "unweighted@0.5": met((gu[S] >= 0.5).astype(int), yS, w), "unweighted@cv": met((gu[S] >= tu).astype(int), yS, w)}
RS = fr.index[(fr.n_tender_fields == 1) & (fr.tenders >= 1) & (fr.tenders <= 100)].to_numpy()
Y = (fr.loc[RS, "tenders"].to_numpy() == 1).astype(float)
cpv = fr.loc[RS, "cpv_div"].fillna("NA"); vc = cpv.value_counts(); cpv = cpv.where(cpv.map(vc) >= 30, "other")
blocks = {"country": pd.get_dummies(fr.loc[RS, "country"].fillna("NA"), prefix="c", drop_first=True), "cpv": pd.get_dummies(cpv, prefix="cpv", drop_first=True),
          "proc": pd.get_dummies(fr.loc[RS, "procedure"].fillna("NA"), prefix="p", drop_first=True), "buyer": pd.get_dummies(fr.loc[RS, "buyer_type"].fillna("NA"), prefix="b", drop_first=True),
          "q": pd.get_dummies(fr.loc[RS, "quarter"], prefix="q", drop_first=True)}
val = pd.to_numeric(fr.loc[RS, "value"], errors="coerce")
vb = pd.DataFrame({"logv": np.log1p(val.fillna(0).clip(lower=0)), "vmiss": val.isna().astype(float)}, index=RS)
posS = {s: k for k, s in enumerate(S)}; lab_rs = np.array([i in posS for i in RS]); pi_rs = fr.loc[RS, "pi"].to_numpy(); x_rs = np.nan_to_num(y[RS])


def design(keys):
    parts = [blocks[k] for k in keys if k != "value"] + ([vb] if "value" in keys else [])
    Wd = pd.concat(parts, axis=1).astype(float) if parts else pd.DataFrame(index=RS)
    Wd["const"] = 1.0; return Wd.to_numpy()


def dsl(g, Wm, Yv=Y, mask=None):
    m = np.ones(len(RS), bool) if mask is None else mask
    gR = g[RS][m]; xt = gR.copy(); lr = lab_rs[m]
    xt[lr] = gR[lr] + (x_rs[m][lr] - gR[lr]) / pi_rs[m][lr]
    Wm = Wm[m]; Yv = Yv[m]; Z = np.column_stack([xt, Wm]); A = Z.T @ Z; A[0, 0] = xt.sum()
    beta = np.linalg.lstsq(A, Z.T @ Yv, rcond=None)[0]
    psi = Z * (Yv - Z @ beta)[:, None]; psi[:, 0] = xt * Yv - xt * beta[0] - xt * (Wm @ beta[1:])
    Ai = np.linalg.pinv(A); se = float(np.sqrt((Ai @ (psi.T @ psi) @ Ai)[0, 0]))
    return {"beta_pp": float(beta[0] * 100), "se_pp": se * 100, "cond": float(np.linalg.cond(A)), "xt_min": float(xt.min()), "xt_max": float(xt.max())}


full = ["country", "cpv", "proc", "buyer", "q", "value"]
diag = {}
for name, keys in [("no_controls", []), ("country_only", ["country"]), ("full", full)]:
    Wm = design(keys)
    for gname, g in [("M5", gw), ("M5u", gu), ("M2_indicator", fr.m2.to_numpy().astype(float))]:
        diag[f"{name}|{gname}"] = dsl(g, Wm)
out["dsl_lpm_diagnosis"] = diag
# 5 single-bidding construct
proc = fr.loc[RS, "procedure"].fillna("NA")
out["single_bid_by_procedure"] = {p: {"n": int((proc == p).sum()), "single_share": float(Y[(proc == p).to_numpy()].mean())} for p in proc.value_counts().index}
op = (proc == "open").to_numpy(); Wf = design(full)


def ols(x, Wm, Yv, mask, wts=None):
    Z = np.column_stack([x[mask], Wm[mask]]); keep = Z.std(0) > 0; keep[0] = True; keep[-1] = True; Z = Z[:, keep]
    wv = np.ones(mask.sum()) if wts is None else wts
    A = (Z * wv[:, None]).T @ Z; b = np.linalg.lstsq(A, (Z * wv[:, None]).T @ Yv[mask], rcond=None)[0]
    e = Yv[mask] - Z @ b; P = Z * (wv * e)[:, None]; Ai = np.linalg.pinv(A); se = float(np.sqrt((Ai @ (P.T @ P) @ Ai)[0, 0]))
    return {"beta_pp": float(b[0] * 100), "ci_pp": [float((b[0] - 1.96 * se) * 100), float((b[0] + 1.96 * se) * 100)], "n": int(mask.sum())}


out["open_procedures_only"] = {"M1_naive": ols(fr.m1.to_numpy()[RS].astype(float), Wf, Y, op), "M2_naive": ols(fr.m2.to_numpy()[RS].astype(float), Wf, Y, op),
                               "labelled_only": ols(x_rs, Wf, Y, op & lab_rs, wts=1 / pi_rs[op & lab_rs]), "DSL_M5": dsl(gw, Wf, mask=op)}
# 6 selection
inR = np.zeros(N, bool); inR[RS] = True
out["selection"] = {"regression": {"n": int(inR.sum()), "M1_share": float(fr.m1[inR].mean()), "M2_share": float(fr.m2[inR].mean()), "share_ENG": float((fr.lang[inR] == "ENG").mean())},
                    "excluded": {"n": int((~inR).sum()), "M1_share": float(fr.m1[~inR].mean()), "M2_share": float(fr.m2[~inR].mean()), "share_ENG": float((fr.lang[~inR] == "ENG").mean())}}
json.dump(out, open(os.path.join(R, "r2_analyses.json"), "w", encoding="utf-8"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))



