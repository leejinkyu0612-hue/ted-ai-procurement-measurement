# -*- coding: utf-8 -*-
"""Step 4: confirmatory analysis exactly as in ANALYSIS_PLAN.md (H1-H4) + pre-specified robustness.
Exploratory analyses are written to a separate block ('exploratory')."""
import os, sys, json, time, glob
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, cohen_kappa_score

BASE = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(BASE, "data"); R = os.path.join(BASE, "results")
SEED = 2027; NB = 2000
fr = pd.read_parquet(os.path.join(D, "frame.parquet"))
N = len(fr)
E = {"e5": np.load(os.path.join(D, "emb_e5small.npy")), "minilm": np.load(os.path.join(D, "emb_minilm.npy"))}

# ---------------- labels ----------------
lab = {}
for line in open(os.path.join(BASE, "logs", "annotation_calls.jsonl"), encoding="utf-8"):
    j = json.loads(line)
    if j.get("label") is not None:
        lab[(j["model"], j["pub"])] = j["label"]
P1, P2 = "anthropic/claude-haiku-5.5", "google/gemini-2.5-flash-lite"
fr["y_p"] = [lab.get((P1, p)) for p in fr.pub]
fr["y_s"] = [lab.get((P2, p)) for p in fr.pub]
nli = pd.read_parquet(os.path.join(D, "nli_scores.parquet")) if os.path.exists(os.path.join(D, "nli_scores.parquet")) else None
fr = fr.merge(nli, on="pub", how="left") if nli is not None else fr.assign(m3_score=np.nan)
m3_full = bool(fr["m3_score"].notna().all())

S = fr.index[(fr.sampled == 1) & fr.y_p.notna()].to_numpy()
miss = int(((fr.sampled == 1) & fr.y_p.isna()).sum())
Nh = fr.groupby("stratum").size()
nh_design = fr[fr.sampled == 1].groupby("stratum").size()
pi_design = (nh_design / Nh).to_dict()
fr["pi_d"] = fr.stratum.map(pi_design)
w = 1.0 / fr.loc[S, "pi_d"].to_numpy()
strata_L = fr.loc[S, "stratum"].to_numpy()
lang_eng = (fr.lang == "ENG").to_numpy()


def labels(defn):
    yp, ys = fr.y_p.to_numpy(dtype=float), fr.y_s.to_numpy(dtype=float)
    if defn == "primary": return yp
    if defn == "and": return np.where(np.isnan(ys), yp, ((yp == 1) & (ys == 1)).astype(float))  # secondary missing -> primary
    if defn == "or": return np.where(np.isnan(ys), yp, ((yp == 1) | (ys == 1)).astype(float))


# ---------------- cross-fitted supervised methods ----------------
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = list(skf.split(S, strata_L))
timing = {}


def fit_lr(X, y, sw):
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y, sample_weight=sw)


def best_thr(score, y, sw):
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(score, np.linspace(0.5, 0.999, 200))):
        p = score >= t
        tp = (sw * (p & (y == 1))).sum(); fp = (sw * (p & (y == 0))).sum(); fn = (sw * (~p & (y == 1))).sum()
        f = 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0
        if f > best: best, bt = f, t
    return bt


tfidf = None


def supervised(y_all, enc="e5"):
    """Returns dict of score arrays over the full frame: cross-fitted on S, full-model elsewhere."""
    global tfidf
    yS = y_all[S].astype(int); out = {}
    X = E[enc]
    t = time.time()
    m5 = np.zeros(N)
    full = fit_lr(X[S], yS, w); m5[:] = full.predict_proba(X)[:, 1]
    for tr, te in folds:
        m = fit_lr(X[S[tr]], yS[tr], w[tr]); m5[S[te]] = m.predict_proba(X[S[te]])[:, 1]
    out["m5"] = m5; timing[f"M5_{enc}_fit_predict_s"] = time.time() - t
    # M4 threshold within training folds
    sc = fr["m4_score" if enc == "e5" else "m4_score_minilm"].to_numpy()
    m4 = (sc >= best_thr(sc[S], yS, w)).astype(int)
    for tr, te in folds:
        th = best_thr(sc[S[tr]], yS[tr], w[tr]); m4[S[te]] = (sc[S[te]] >= th).astype(int)
    out["m4"] = m4; out["m4_score"] = sc
    if enc == "e5":
        t = time.time()
        if tfidf is None:
            tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True).fit(fr.text.iloc[S])
        T = tfidf.transform(fr.text)
        m6 = np.zeros(N); full6 = fit_lr(T[S], yS, w); m6[:] = full6.predict_proba(T)[:, 1]
        for tr, te in folds:
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=200000, sublinear_tf=True).fit(fr.text.iloc[S[tr]])
            m = fit_lr(vec.transform(fr.text.iloc[S[tr]]), yS[tr], w[tr]); m6[S[te]] = m.predict_proba(vec.transform(fr.text.iloc[S[te]]))[:, 1]
        out["m6"] = m6; timing["M6_tfidf_fit_predict_s"] = time.time() - t
    return out


def binarize(sc, thr=0.5):
    return (np.nan_to_num(sc, nan=-1) >= thr).astype(int)


def metrics(pred, y, ww, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    p, yy, sw = pred[m] == 1, y[m] == 1, ww[m]
    tp = (sw * (p & yy)).sum(); fp = (sw * (p & ~yy)).sum(); fn = (sw * (~p & yy)).sum()
    prec = tp / (tp + fp) if tp + fp > 0 else np.nan; rec = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else np.nan
    return prec, rec, f1


rng = np.random.default_rng(SEED)
boot_idx = []
for b in range(NB):
    ii = []
    for h in np.unique(strata_L):
        loc = np.where(strata_L == h)[0]; ii.append(rng.choice(loc, len(loc), replace=True))
    boot_idx.append(np.concatenate(ii))


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def evaluate(preds, y_all, label_name, thr_note=""):
    yS = y_all[S]; res = {}
    for k, pv in preds.items():
        pS = pv[S]
        pr, rc, f1 = metrics(pS, yS, w)
        bs = [metrics(pS[bi], yS[bi], w[bi]) for bi in boot_idx]
        res[k] = {"precision": pr, "recall": rc, "f1": f1, "precision_ci": ci([b[0] for b in bs]),
                  "recall_ci": ci([b[1] for b in bs]), "f1_ci": ci([b[2] for b in bs]),
                  "recall_eng": metrics(pS, yS, w, lang_eng[S])[1], "recall_noneng": metrics(pS, yS, w, ~lang_eng[S])[1]}
    return res


def dsl_prev(g, y_all, idx_boot=True):
    corr = w * (y_all[S] - g[S])
    est = (g.sum() + corr.sum()) / N
    bs = [(g.sum() + (w[bi] * (y_all[S][bi] - g[S][bi])).sum()) / N for bi in boot_idx] if idx_boot else []
    return est, bs


# ---------------- regression sample and design ----------------
RS = fr.index[(fr.n_tender_fields == 1) & (fr.tenders >= 1) & (fr.tenders <= 100)].to_numpy()
Y = (fr.loc[RS, "tenders"].to_numpy() == 1).astype(float)
cpv = fr.loc[RS, "cpv_div"].fillna("NA"); vc = cpv.value_counts(); cpv = cpv.where(cpv.map(vc) >= 30, "other")
Wdf = pd.concat([pd.get_dummies(fr.loc[RS, "country"].fillna("NA"), prefix="c", drop_first=True),
                 pd.get_dummies(cpv, prefix="cpv", drop_first=True),
                 pd.get_dummies(fr.loc[RS, "procedure"].fillna("NA"), prefix="proc", drop_first=True),
                 pd.get_dummies(fr.loc[RS, "buyer_type"].fillna("NA"), prefix="bt", drop_first=True),
                 pd.get_dummies(fr.loc[RS, "quarter"], prefix="q", drop_first=True)], axis=1).astype(float)
val = pd.to_numeric(fr.loc[RS, "value"], errors="coerce")
Wdf["logv"] = np.log1p(val.fillna(0).clip(lower=0)); Wdf["v_missing"] = val.isna().astype(float)
Wdf["const"] = 1.0
Wm = Wdf.to_numpy()
Wm = Wm[:, Wm.std(0) > 0 if True else slice(None)] if False else Wm
inRS = np.zeros(N, bool); inRS[RS] = True
posS = {s: i for i, s in enumerate(S)}


def ols_robust(X, Yv, wts=None):
    Z = np.column_stack([X, Wm]) if wts is None else X
    wv = np.ones(len(Yv)) if wts is None else wts
    A = (Z * wv[:, None]).T @ Z; b = (Z * wv[:, None]).T @ Yv
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    e = Yv - Z @ beta; psi = Z * (wv * e)[:, None]
    Ai = np.linalg.pinv(A); V = Ai @ (psi.T @ psi) @ Ai
    return float(beta[0]), float(np.sqrt(V[0, 0]))


def dsl_lpm(g, y_all):
    """X~ = g + R/pi (X - g); OLS moments with X^2 replaced by X~ (X binary)."""
    gR = g[RS]; Rr = np.array([i in posS for i in RS]); xt = gR.copy()
    for k, i in enumerate(RS):
        if i in posS:
            xt[k] = gR[k] + (y_all[i] - gR[k]) / fr.at[i, "pi_d"]
    Z = np.column_stack([xt, Wm])
    A = Z.T @ Z; A[0, 0] = xt.sum()
    b = Z.T @ Y
    beta = np.linalg.solve(A + 1e-9 * np.eye(len(A)), b)
    # per-observation moments: z_i (y_i - z_i'beta) with x_i^2 -> xt_i
    pred = Z @ beta; resid = Y - pred
    psi = Z * resid[:, None]
    psi[:, 0] = xt * Y - (xt * beta[0] + xt * (Wm @ beta[1:]))
    Ai = np.linalg.pinv(A); V = Ai @ (psi.T @ psi) @ Ai
    return float(beta[0]), float(np.sqrt(V[0, 0])), int(Rr.sum())


def labeled_only(y_all):
    idx = np.array([i for i in S if inRS[i]])
    k = np.array([np.where(RS == i)[0][0] for i in idx])
    Z = np.column_stack([y_all[idx], Wm[k]])
    keep = Z.std(0) > 0; keep[0] = True; keep[-1] = True
    Z = Z[:, keep]
    wv = 1.0 / fr.loc[idx, "pi_d"].to_numpy()
    A = (Z * wv[:, None]).T @ Z; b = (Z * wv[:, None]).T @ Y[k]
    beta = np.linalg.lstsq(A, b, rcond=None)[0]; e = Y[k] - Z @ beta
    psi = Z * (wv * e)[:, None]; Ai = np.linalg.pinv(A); V = Ai @ (psi.T @ psi) @ Ai
    return float(beta[0]), float(np.sqrt(V[0, 0])), int(len(idx))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.zeros(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj.tolist()


def run_all(defn, enc="e5", full=True):
    y_all = labels(defn)
    sup = supervised(y_all, enc)
    preds = {"M1_en_dict": fr.m1.to_numpy(), "M2_ml_dict": fr.m2.to_numpy(),
             "M4_proto_sim": sup["m4"], "M5_emb_lr": binarize(sup["m5"]), }
    if enc == "e5":
        preds["M3_nli"] = binarize(fr.m3_score.to_numpy()); preds["M6_tfidf_lr"] = binarize(sup["m6"])
    ev = evaluate(preds, y_all, defn)
    out = {"label_def": defn, "encoder": enc, "n_labeled": int(len(S)), "metrics": ev}
    # AUPRC for score methods
    yS = y_all[S].astype(int)
    out["auprc"] = {k: float(average_precision_score(yS, v[S], sample_weight=w)) for k, v in
                    [("M4_proto_sim", sup["m4_score"]), ("M5_emb_lr", sup["m5"])] + ([("M3_nli", np.nan_to_num(fr.m3_score.to_numpy())), ("M6_tfidf_lr", sup["m6"])] if enc == "e5" else [])}
    if not full:
        return out, sup
    # H1, H2
    pS = {k: v[S] for k, v in preds.items()}
    yS_f = y_all[S]
    d1 = [metrics(pS["M1_en_dict"][bi], yS_f[bi], w[bi], ~lang_eng[S][bi])[1] - metrics(pS["M1_en_dict"][bi], yS_f[bi], w[bi], lang_eng[S][bi])[1] for bi in boot_idx]
    obs1 = ev["M1_en_dict"]["recall_noneng"] - ev["M1_en_dict"]["recall_eng"]
    d2a = [metrics(pS["M5_emb_lr"][bi], yS_f[bi], w[bi])[2] - metrics(pS["M1_en_dict"][bi], yS_f[bi], w[bi])[2] for bi in boot_idx]
    d2b = [metrics(pS["M5_emb_lr"][bi], yS_f[bi], w[bi])[2] - metrics(pS["M2_ml_dict"][bi], yS_f[bi], w[bi])[2] for bi in boot_idx]
    # prevalence
    naive = {k: float(v.mean()) for k, v in preds.items()}
    ht = float((w * yS_f).sum() / N)
    dsl, dsl_bs = dsl_prev(sup["m5"], y_all)
    d3 = [b - naive["M1_en_dict"] for b in dsl_bs]
    p1 = (np.sum(np.array(d1) >= 0) + 1) / (NB + 1); p2a = (np.sum(np.array(d2a) <= 0) + 1) / (NB + 1)
    p2b = (np.sum(np.array(d2b) <= 0) + 1) / (NB + 1); p3 = (np.sum(np.array(d3) <= 0) + 1) / (NB + 1)
    out["H1"] = {"diff_recall_noneng_minus_eng": obs1, "ci": ci(d1), "p_one_sided": p1}
    out["H2a"] = {"f1_M5_minus_M1": ev["M5_emb_lr"]["f1"] - ev["M1_en_dict"]["f1"], "ci": ci(d2a), "p_one_sided": p2a}
    out["H2b"] = {"f1_M5_minus_M2": ev["M5_emb_lr"]["f1"] - ev["M2_ml_dict"]["f1"], "ci": ci(d2b), "p_one_sided": p2b}
    out["prevalence"] = {"N_frame": N, "naive": naive, "ht_labels_only": ht, "ht_ci": ci([(w[bi] * yS_f[bi]).sum() / N for bi in boot_idx]),
                         "dsl": dsl, "dsl_ci": ci(dsl_bs)}
    out["H3"] = {"dsl_minus_M1": dsl - naive["M1_en_dict"], "ci": ci(d3), "p_one_sided": p3}
    out["holm_adjusted"] = dict(zip(["H1", "H2a", "H2b", "H3"], holm([p1, p2a, p2b, p3])))
    # H4
    reg = {"n_regression": int(len(RS)), "single_bid_share": float(Y.mean())}
    for k, v in preds.items():
        b_, se_ = ols_robust(v[RS].astype(float), Y); reg[f"naive_{k}"] = {"beta": b_, "se": se_, "ci": [b_ - 1.96 * se_, b_ + 1.96 * se_], "n_ai": int(v[RS].sum())}
    b_, se_, nl = dsl_lpm(sup["m5"], y_all); reg["DSL_M5"] = {"beta": b_, "se": se_, "ci": [b_ - 1.96 * se_, b_ + 1.96 * se_], "n_labeled_in_reg": nl}
    b_, se_, nl = labeled_only(y_all); reg["labeled_only"] = {"beta": b_, "se": se_, "ci": [b_ - 1.96 * se_, b_ + 1.96 * se_], "n": nl}
    out["H4"] = reg
    return out, sup


results = {"plan_commit": "7314e8ddc14fb973391f1062c204b0a7f4642e99", "missing_primary_labels_in_sample": miss,
           "strata": {h: {"N": int(Nh[h]), "n": int(nh_design.get(h, 0)), "pi": float(pi_design[h])} for h in Nh.index},
           "m3_full_frame": m3_full}
main, sup_main = run_all("primary")
results["main"] = main
# annotator agreement
both = (fr.sampled == 1) & fr.y_p.notna() & fr.y_s.notna()
wb = 1 / fr.loc[both, "pi_d"]
results["agreement"] = {"n_both": int(both.sum()), "kappa_weighted": float(cohen_kappa_score(fr.loc[both, "y_p"], fr.loc[both, "y_s"], sample_weight=wb)),
                        "kappa_unweighted": float(cohen_kappa_score(fr.loc[both, "y_p"], fr.loc[both, "y_s"])),
                        "raw_agreement": float((fr.loc[both, "y_p"] == fr.loc[both, "y_s"]).mean()),
                        "pos_primary": int(fr.loc[both, "y_p"].sum()), "pos_secondary": int(fr.loc[both, "y_s"].sum())}
for g, msk in [("ENG", fr.lang == "ENG"), ("nonENG", fr.lang != "ENG")]:
    bb = both & msk
    if bb.sum() > 10:
        results["agreement"][f"kappa_{g}"] = float(cohen_kappa_score(fr.loc[bb, "y_p"], fr.loc[bb, "y_s"], sample_weight=1 / fr.loc[bb, "pi_d"]))
        results["agreement"][f"n_{g}"] = int(bb.sum())
# robustness
rob = {}
for defn in ["and", "or"]:
    r_, _ = run_all(defn)
    rob[f"label_{defn}"] = {k: r_[k] for k in ["metrics", "H1", "H2a", "H2b", "prevalence", "H3", "H4"]}
r_, _ = run_all("primary", enc="minilm")
rob["encoder_minilm"] = {k: r_[k] for k in ["metrics", "H1", "H2a", "H2b", "prevalence", "H3", "H4"]}
y_all = labels("primary"); yS = y_all[S]
thr_rob = {}
for t in [0.3, 0.7]:
    for k, sc in [("M3_nli", fr.m3_score.to_numpy()), ("M5_emb_lr", sup_main["m5"]), ("M6_tfidf_lr", sup_main["m6"])]:
        pr, rc, f1 = metrics(binarize(sc, t)[S], yS, w); thr_rob[f"{k}@{t}"] = {"precision": pr, "recall": rc, "f1": f1}
rob["thresholds"] = thr_rob
ne = ~lang_eng[S]
rob["H2_nonENG_only"] = {k: metrics(v[S], yS, w, ne)[2] for k, v in
                         [("M1_en_dict", fr.m1.to_numpy()), ("M2_ml_dict", fr.m2.to_numpy()), ("M5_emb_lr", binarize(sup_main["m5"]))]}
# logistic plug-in (H4b robustness)
import statsmodels.api as sm
logit = {}
for k, v in [("M1_en_dict", fr.m1.to_numpy()), ("M2_ml_dict", fr.m2.to_numpy()), ("M5_emb_lr", binarize(sup_main["m5"]))]:
    Z = np.column_stack([v[RS].astype(float), Wm])
    keep = Z.std(0) > 0; keep[0] = True; keep[-1] = True
    try:
        mfit = sm.GLM(Y, Z[:, keep], family=sm.families.Binomial()).fit(cov_type="HC1")
        logit[k] = {"coef": float(mfit.params[0]), "se": float(mfit.bse[0]), "odds_ratio": float(np.exp(mfit.params[0]))}
    except Exception as e:
        logit[k] = {"error": str(e)[:200]}
rob["H4b_logit_plugin"] = logit
results["robustness"] = rob
results["timing_s"] = timing
for f in ["step1_sample.json", "step3_nli_timing.json"]:
    p = os.path.join(R, f)
    if os.path.exists(p): results[f.replace(".json", "")] = json.load(open(p))

# ---------------- exploratory ----------------
exp = {}
per_lang = {}
for L_ in fr.loc[S, "lang"].value_counts().index:
    msk = (fr.loc[S, "lang"] == L_).to_numpy()
    npos = int((yS[msk] == 1).sum())
    if npos >= 5:
        per_lang[L_] = {"n_labeled": int(msk.sum()), "n_pos": npos,
                        **{k: metrics(v[S], yS, w, msk)[1] for k, v in [("M1", fr.m1.to_numpy()), ("M2", fr.m2.to_numpy()), ("M5", binarize(sup_main["m5"]))]}}
exp["recall_by_language"] = per_lang
# country prevalence ranking: DSL-style per country vs M1
g = sup_main["m5"]; xt = g.copy()
for i in S: xt[i] = g[i] + (y_all[i] - g[i]) / fr.at[i, "pi_d"]
fr["_xt"] = xt
cn = fr.groupby("country").agg(n=("pub", "size"), m1=("m1", "mean"), dsl=("_xt", "mean"))
cn = cn[cn.n >= 500]
from scipy.stats import spearmanr
exp["country_rank"] = {"n_countries": int(len(cn)), "spearman_m1_vs_dsl": float(spearmanr(cn.m1, cn.dsl).statistic),
                       "table": cn.sort_values("dsl", ascending=False).round(4).reset_index().to_dict("records")}
results["exploratory"] = exp
json.dump(results, open(os.path.join(R, "analysis_main.json"), "w"), indent=1, default=float)
fr[["pub", "lang", "country", "stratum", "sampled", "pi_d", "y_p", "y_s", "m1", "m2", "m3_score", "m4_score"]].assign(
    m5_prob=sup_main["m5"], m6_prob=sup_main["m6"]).to_csv(os.path.join(R, "notice_level_scores.csv.gz"), index=False, compression="gzip")
print(json.dumps({k: results["main"][k] for k in ["H1", "H2a", "H2b", "H3", "holm_adjusted"]}, indent=1, default=float))
print(json.dumps(results["main"]["prevalence"], indent=1, default=float))
print(json.dumps({k: v for k, v in results["main"]["H4"].items()}, indent=1, default=float))
print(json.dumps(results["agreement"], indent=1))

