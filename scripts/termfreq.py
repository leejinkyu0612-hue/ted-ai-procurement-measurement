# -*- coding: utf-8 -*-
"""S13: count in how many prior-study PDFs (S02_선행연구/papers, 22 papers) each candidate term / variant occurs."""
import os, re, glob, json
import pypdf
P = os.path.join(os.path.dirname(__file__), "..", "..", "S02_선행연구", "papers")
texts = {}
for f in sorted(glob.glob(os.path.join(P, "*.pdf"))):
    try:
        r = pypdf.PdfReader(f); t = " ".join((p.extract_text() or "") for p in r.pages[:40])
    except Exception:
        t = ""
    texts[os.path.basename(f)] = re.sub(r"\s+", " ", t.replace("-\n", "")).lower()
groups = {
 "AI-related / AI-relatedness": ["ai-related", "ai-relatedness", "ai relatedness", "ai-related notices"],
 "public procurement": ["public procurement", "government procurement", "public purchasing"],
 "contract award notice": ["contract award notice", "award notice", "contract notice", "call for tenders", "tender notice"],
 "single bidding": ["single bidding", "single-bidding", "single bidder", "single-bid", "one bidder", "number of bidders", "number of offers"],
 "keyword dictionary": ["dictionary", "keyword list", "keyword", "lexicon", "dictionary-based"],
 "multilingual": ["multilingual", "cross-lingual", "multi-lingual"],
 "sentence embedding": ["sentence embedding", "text embedding", "embeddings", "sentence encoder"],
 "zero-shot classification": ["zero-shot", "zero shot"],
 "natural language inference": ["natural language inference", "nli", "entailment"],
 "large language model": ["large language model", "llm"],
 "annotation": ["annotation", "annotator", "labeling", "labelling", "coding"],
 "reference labels / gold standard": ["gold standard", "gold-standard", "ground truth", "expert annotation", "reference label"],
 "measurement error": ["measurement error", "misclassification", "prediction error"],
 "prevalence": ["prevalence", "proportion", "share of"],
 "design-based supervised learning": ["design-based supervised learning", "dsl"],
 "prediction-powered inference": ["prediction-powered inference", "ppi"],
 "Horvitz-Thompson / inverse probability weighting": ["horvitz", "inverse probability", "sampling weight", "design weight"],
 "stratified sampling": ["stratified", "stratum", "strata"],
 "probability sample": ["probability sample", "random sample", "sampling design"],
 "cross-fitting": ["cross-fitting", "cross-fit", "cross fitting", "sample splitting"],
 "precision / recall / F1": ["precision", "recall", "f1"],
 "AUPRC / average precision": ["auprc", "average precision", "precision-recall curve"],
 "Cohen's kappa": ["kappa", "inter-annotator", "intercoder", "inter-coder", "agreement"],
 "linear probability model": ["linear probability model", "linear regression", "ols"],
 "fixed effects": ["fixed effect", "fixed-effect"],
 "downstream inference": ["downstream", "second-stage", "two-step"],
 "bootstrap confidence interval": ["bootstrap"],
 "pre-specified / pre-registered": ["pre-specified", "prespecified", "pre-registered", "preregistered", "pre-analysis plan"],
 "post hoc / exploratory": ["post hoc", "post-hoc", "exploratory"],
 "TED / Tenders Electronic Daily": ["tenders electronic daily", " ted "],
 "CPV": ["cpv", "common procurement vocabulary"],
 "competition": ["competition", "competitive"],
 "robustness / sensitivity": ["robustness", "sensitivity analysis"],
}
rows = []
for g, vars_ in groups.items():
    hit = lambda v, t: (re.search(r"\b" + re.escape(v.strip()) + r"s?\b", t) is not None) if len(v.strip()) <= 5 else (v in t)
    per = {v: sum(1 for t in texts.values() if hit(v, t)) for v in vars_}
    any_ = sum(1 for t in texts.values() if any(hit(v, t) for v in vars_))
    rows.append({"group": g, "variants": per, "papers_any": any_})
json.dump({"n_papers": len(texts), "papers": list(texts), "rows": rows}, open(os.path.join(os.path.dirname(__file__), "..", "..", "S13_용어변경", "선행연구_용어빈도.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in rows: print(r["group"], r["papers_any"], r["variants"])

