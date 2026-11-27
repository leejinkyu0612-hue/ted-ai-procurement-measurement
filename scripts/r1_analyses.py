# -*- coding: utf-8 -*-
"""R1 analyses (added after ChatGPT review R1; all POST HOC unless they only re-check pre-specified quantities).
1 realized inclusion probabilities (one unparsable primary label in stratum C)
2 annotator agreement with recovered secondary labels: kappa (bootstrap CI), positive/negative agreement, failure rates
3 adjudicated majority reference X_maj (primary, secondary, deepseek adjudicator for disagreements and C/D positives)
4 leave-one-out influence of every C/D positive on prevalence, H3 and M1/M2 recall
5 DSL-LPM implementation check: simulation with known beta (coverage), condition number, bootstrap CI
6 HT vs DSL standard errors; regression-sample flow counts
"""
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
BASE = os.path.join(os.path.dirname(__file__), ".."); D = os.path.join(BASE, "data"); R = os.path.join(BASE, "results")
SEED = 2027; NB = 2000
fr = pd.read_parquet(os.path.join(D, "frame.parquet")); N = len(fr); E = np.load(os.path.join(D, "emb_e5small.npy"))
lab = {}
for line in open(os.path.join(BASE, "logs", "annotation_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is not None: lab[(j["model"], j["pub"])] = j["label"]
rec_s, adj = {}, {}
for line in open(os.path.join(BASE, "logs", "r1_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is None: continue
    (rec_s if j["task"] == "reparse_secondary" else adj)[j["pub"]] = j["label"]
fr["y_p"] = [lab.get(("anthropic/claude-haiku-5.5", p)) for p in fr.pub]
fr["y_s0"] = [lab.get(("google/gemini-2.5-flash-lite", p)) for p in fr.pub]
fr["y_s"] = [a if not pd.isna(a) else rec_s.get(p) for a, p in zip(fr.y_s0, fr.pub)]
fr["y_a"] = [adj.get(p) for p in fr.pub]
S = fr.index[(fr.sampled == 1) & fr.y_p.notna()].to_numpy()
Nh = fr.groupby("stratum").size()
nh_real = fr.loc[S].groupby("stratum").size()
fr["pi_r"] = fr.stratum.map((nh_real / Nh).to_dict())
w = 1 / fr.loc[S, "pi_r"].to_numpy(); st = fr.loc[S, "stratum"].to_numpy(); eng = (fr.lang == "ENG").to_numpy()
rng = np.random.default_rng(SEED)
boot = [np.concatenate([rng.choice(np.where(st == h)[0], (st == h).sum(), replace=True) for h in np.unique(st)]) for _ in range(NB)]
out = {"note": "R1 analyses, post hoc", "pi_realized": {h: float(nh_real[h] / Nh[h]) for h in Nh.index}}


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]; return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def met(p, y, ww, m=None):
    m = np.ones(len(y), bool) if m is None else m
    p, y, ww = p[m] == 1, y[m] == 1, ww[m]
    tp = (ww * (p & y)).sum(); fp = (ww * (p & ~y)).sum(); fn = (ww * (~p & y)).sum()
    return (tp / (tp + fp) if tp + fp else np.nan, tp / (tp + fn) if tp + fn else np.nan, 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)


# 2 agreement
L = fr.loc[S]
both = L.y_s.notna()
yp, ys, wb = L.y_p[both].to_numpy(int), L.y_s[both].to_numpy(int), (1 / L.pi_r[both]).to_numpy()
stb = L.stratum[both].to_numpy()
kb = [cohen_kappa_score(yp[b], ys[b], sample_weight=wb[b]) for b in
      [np.concatenate([rng.choice(np.where(stb == h)[0], (stb == h).sum(), replace=True) for h in np.unique(stb)]) for _ in range(500)]]
pa = 2 * (wb * (yp & ys)).sum() / (wb * (yp + ys)).sum(); na = 2 * (wb * ((1 - yp) & (1 - ys))).sum() / (wb * ((1 - yp) + (1 - ys))).sum()
out["agreement"] = {"n_secondary_original": int(L.y_s0.notna().sum()), "n_secondary_recovered": int(L.y_s.notna().sum() - L.y_s0.notna().sum()),
                    "n_both": int(both.sum()), "kappa_w": float(cohen_kappa_score(yp, ys, sample_weight=wb)), "kappa_w_ci": ci(kb),
                    "kappa_unw": float(cohen_kappa_score(yp, ys)), "pos_agreement_w": float(pa), "neg_agreement_w": float(na),
                    "kappa_w_ENG": float(cohen_kappa_score(yp[eng[S][both]], ys[eng[S][both]], sample_weight=wb[eng[S][both]])),
                    "kappa_w_nonENG": float(cohen_kappa_score(yp[~eng[S][both]], ys[~eng[S][both]], sample_weight=wb[~eng[S][both]])),
                    "secondary_missing_by_stratum": L[L.y_s.isna()].stratum.value_counts().to_dict(),
                    "secondary_missing_by_lang_top": L[L.y_s.isna()].lang.value_counts().head(6).to_dict()}
# 3 majority reference
def maj(r):
    v = [r.y_p] + ([r.y_s] if pd.notna(r.y_s) else [])
    if len(v) == 2 and v[0] == v[1]: return v[0]
    if pd.notna(r.y_a): v.append(r.y_a)
    if len(v) == 1: return v[0]
    s = sum(v); return 1 if s * 2 > len(v) else (0 if s * 2 < len(v) else r.y_a if pd.notna(r.y_a) else r.y_p)
fr["y_m"] = np.nan; fr.loc[S, "y_m"] = [maj(r) for r in fr.loc[S].itertuples()]
cd = fr.loc[S][(fr.loc[S].stratum.isin(["C", "D"])) & (fr.loc[S].y_p == 1)]
out["CD_positives"] = [{"pub": r.pub, "stratum": r.stratum, "lang": r.lang, "primary": r.y_p, "secondary": None if pd.isna(r.y_s) else int(r.y_s),
                        "adjudicator": None if pd.isna(r.y_a) else int(r.y_a), "majority": int(r.y_m), "text": r.text[:200]} for r in cd.itertuples()]
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(S, st))


def g_cross(y):
    yS = y[S].astype(int); g = LogisticRegression(C=1.0, max_iter=3000).fit(E[S], yS, sample_weight=w).predict_proba(E)[:, 1]
    for tr, te in folds:
        g[S[te]] = LogisticRegression(C=1.0, max_iter=3000).fit(E[S[tr]], yS[tr], sample_weight=w[tr]).predict_proba(E[S[te]])[:, 1]
    return g


RS = fr.index[(fr.n_tender_fields == 1) & (fr.tenders >= 1) & (fr.tenders <= 100)].to_numpy()
Y = (fr.loc[RS, "tenders"].to_numpy() == 1).astype(float)
cpv = fr.loc[RS, "cpv_div"].fillna("NA"); vc = cpv.value_counts(); cpv = cpv.where(cpv.map(vc) >= 30, "other")
Wd = pd.concat([pd.get_dummies(fr.loc[RS, "country"].fillna("NA"), prefix="c", drop_first=True), pd.get_dummies(cpv, prefix="cpv", drop_first=True),
                pd.get_dummies(fr.loc[RS, "procedure"].fillna("NA"), prefix="p", drop_first=True), pd.get_dummies(fr.loc[RS, "buyer_type"].fillna("NA"), prefix="b", drop_first=True),
                pd.get_dummies(fr.loc[RS, "quarter"], prefix="q", drop_first=True)], axis=1).astype(float)
val = pd.to_numeric(fr.loc[RS, "value"], errors="coerce"); Wd["logv"] = np.log1p(val.fillna(0).clip(lower=0)); Wd["vmiss"] = val.isna().astype(float); Wd["const"] = 1.0
Wm = Wd.to_numpy(); posS = {s: k for k, s in enumerate(S)}
lab_in_rs = np.array([i in posS for i in RS]); pi_rs = fr.loc[RS, "pi_r"].to_numpy()


def dsl_lpm(g, x_true, Yv=Y, R_mask=lab_in_rs, pis=pi_rs):
    gR = g[RS] if len(g) == N else g
    xt = gR.copy(); xt[R_mask] = gR[R_mask] + (x_true[R_mask] - gR[R_mask]) / pis[R_mask]
    Z = np.column_stack([xt, Wm]); A = Z.T @ Z; A[0, 0] = xt.sum(); beta = np.linalg.solve(A, Z.T @ Yv)
    psi = Z * (Yv - Z @ beta)[:, None]; psi[:, 0] = xt * Yv - xt * beta[0] - xt * (Wm @ beta[1:])
    Ai = np.linalg.inv(A); V = Ai @ (psi.T @ psi) @ Ai
    return float(beta[0]), float(np.sqrt(V[0, 0])), float(np.linalg.cond(A))


def summary(y, name):
    yS = y[S]; g = g_cross(y)
    res = {}
    for k, p in [("M1", fr.m1.to_numpy()), ("M2", fr.m2.to_numpy())]:
        pr, rc, f1 = met(p[S], yS, w); res[k] = {"precision": pr, "recall": rc, "f1": f1, "f1_ci": ci([met(p[S][b], yS[b], w[b])[2] for b in boot]),
                                                 "recall_eng": met(p[S], yS, w, eng[S])[1], "recall_noneng": met(p[S], yS, w, ~eng[S])[1]}
    d1 = [met(fr.m1.to_numpy()[S][b], yS[b], w[b], ~eng[S][b])[1] - met(fr.m1.to_numpy()[S][b], yS[b], w[b], eng[S][b])[1] for b in boot]
    ht = (w * yS).sum() / N; dsl = (g.sum() + (w * (yS - g[S])).sum()) / N
    dbs = [(g.sum() + (w[b] * (yS[b] - g[S][b])).sum()) / N for b in boot]
    xfull = np.zeros(N); xfull[S] = yS
    b_, se_, cond = dsl_lpm(g, xfull[RS])
    res.update({"H1_diff": res["M1"]["recall_noneng"] - res["M1"]["recall_eng"], "H1_ci": ci(d1), "HT": float(ht), "HT_ci": ci([(w[b] * yS[b]).sum() / N for b in boot]),
                "HT_se_boot": float(np.std([(w[b] * yS[b]).sum() / N for b in boot])), "DSL": float(dsl), "DSL_ci": ci(dbs), "DSL_se_boot": float(np.std(dbs)),
                "H3_diff": float(dsl - fr.m1.mean()), "H3_ci": ci([x - fr.m1.mean() for x in dbs]),
                "DSL_beta": b_, "DSL_se": se_, "DSL_beta_ci": [b_ - 1.96 * se_, b_ + 1.96 * se_], "cond_A": cond, "n_pos": int(yS.sum())})
    return res, g


out["primary_realized_pi"], g_p = summary(fr.y_p.to_numpy(dtype=float), "primary")
out["majority_reference"], g_m = summary(fr.y_m.to_numpy(dtype=float), "majority")
# bootstrap CI of DSL beta (primary), labeled resampling, g fixed
y_p = fr.y_p.to_numpy(dtype=float); xfull = np.zeros(N); xfull[S] = y_p[S]
bb = []
for b in boot[:500]:
    cnt = np.bincount(b, minlength=len(S))  # multiplicity of each labeled unit
    mult = np.zeros(N); mult[S] = cnt
    gR = g_p[RS]; xt = gR.copy(); m = lab_in_rs
    xt[m] = gR[m] + mult[RS][m] * (xfull[RS][m] - gR[m]) / pi_rs[m]
    Z = np.column_stack([xt, Wm]); A = Z.T @ Z; A[0, 0] = xt.sum()
    try: bb.append(np.linalg.solve(A, Z.T @ Y)[0])
    except Exception: pass
out["primary_realized_pi"]["DSL_beta_boot_ci"] = ci(bb)
# 4 leave-one-out for C/D positives (primary reference)
loo = []
yS = y_p[S]
for r in cd.itertuples():
    k = posS[r.Index]; y2 = yS.copy(); y2[k] = 0
    ht = (w * y2).sum() / N; dsl = (g_p.sum() + (w * (y2 - g_p[S])).sum()) / N
    loo.append({"pub": r.pub, "stratum": r.stratum, "HT_without": float(ht), "DSL_without": float(dsl), "H3_without": float(dsl - fr.m1.mean()),
                "M1_recall_without": met(fr.m1.to_numpy()[S], y2, w)[1], "M2_recall_without": met(fr.m2.to_numpy()[S], y2, w)[1]})
out["loo_CD_positives"] = loo
# 5 simulation check of DSL-LPM (known beta)
srng = np.random.default_rng(1)
q = np.clip(g_m[RS] * 0.9 + 0.002, 0, 1)
sim = []
strat_rs = fr.loc[RS, "stratum"].to_numpy()
for rep in range(200):
    X = (srng.random(len(RS)) < q).astype(float)
    Ys = (srng.random(len(RS)) < np.clip(0.46 + 0.10 * X, 0, 1)).astype(float)
    Rm = np.zeros(len(RS), bool); pis = np.zeros(len(RS))
    for h in ["A", "B", "C", "D"]:
        idx = np.where(strat_rs == h)[0]; n = max(1, int(round(len(idx) * float(out["pi_realized"][h]))))
        Rm[srng.choice(idx, n, replace=False)] = True; pis[idx] = n / len(idx)
    gnoisy = np.clip(q * 0.3, 1e-4, 1)  # deliberately miscalibrated predictor
    b_, se_, _ = dsl_lpm(gnoisy, X, Yv=Ys, R_mask=Rm, pis=pis)
    sim.append((b_, se_))
sim = np.array(sim)
out["dsl_simulation"] = {"true_beta": 0.10, "reps": 200, "mean_beta": float(sim[:, 0].mean()), "median_beta": float(np.median(sim[:, 0])),
                         "coverage_95": float(np.mean(np.abs(sim[:, 0] - 0.10) <= 1.96 * sim[:, 1])), "mean_se": float(sim[:, 1].mean()),
                         "note": "X simulated from majority-reference probabilities, same strata and realized pi, predictor deliberately miscalibrated (0.3*q)"}
# 6 regression sample flow
fl = {"frame": N, "exactly_one_tender_field": int((fr.n_tender_fields == 1).sum()), "no_tender_field": int((fr.n_tender_fields == 0).sum()),
      "multi_tender_fields": int((fr.n_tender_fields > 1).sum()), "one_field_tenders_0": int(((fr.n_tender_fields == 1) & (fr.tenders == 0)).sum()),
      "one_field_tenders_gt100": int(((fr.n_tender_fields == 1) & (fr.tenders > 100)).sum()), "regression_sample": int(len(RS)),
      "labeled_in_regression": int(lab_in_rs.sum())}
out["regression_flow"] = fl
# 7 X_and / X_or recomputed with recovered secondary labels (missing secondary -> primary)
ypa = fr.y_p.to_numpy(dtype=float); ysa = fr.y_s.to_numpy(dtype=float)
x_and = np.where(np.isnan(ysa), ypa, ((ypa == 1) & (ysa == 1)).astype(float)); x_or = np.where(np.isnan(ysa), ypa, ((ypa == 1) | (ysa == 1)).astype(float))
out["and_recovered"], _ = summary(x_and, "and"); out["or_recovered"], _ = summary(x_or, "or")
out["n_cpv_levels_used"] = int(cpv.nunique())
json.dump(out, open(os.path.join(R, "r1_analyses.json"), "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items() if k not in ("CD_positives",)}, indent=1, default=float, ensure_ascii=False)[:9000])
for c in out["CD_positives"]: print(c)




