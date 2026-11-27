# Identifying AI-related notices in multilingual public procurement

Code, prompts, dictionaries, analysis plan, annotation logs, and notice-level derived data for the paper
"Identifying AI-Related IT and Software Award Notices in Tenders Electronic Daily: Keyword Dictionaries, Embedding Classifiers, and Design-Based Supervised Learning"
(Jinkyu Lee and Chang Soo Sung, Dongguk University; submitted to AI-Biz2027 @ JSAI-isAI 2027).

## Pre-specification
- `ANALYSIS_PLAN.md` — fixed before reference annotation (commit `7314e8d`, 2026-11-27 11:12 KST).
- `AMENDMENT_1.md` — committed after annotation ended (12:16 KST) but before any M3 result (commit `b1eb797`, 12:22 KST).
- Everything added later (scripts `r1_*`, `r2_*`, `r3_*`, `step5_exploratory.py`) is post hoc / exploratory.

## Data
- Source: TED Search API v3 (anonymous), contract award notices (`can-standard`) with CPV 72000000 or 48000000, published 2024-01-01..2025-12-31.
  TED notices may be reused for commercial and non-commercial purposes (TED legal notice; Commission Decision 2011/833/EU). Source: (c) European Union, TED.
- Raw notice texts are **not** redistributed. `scripts/fetch_ted.py` re-downloads them; `scripts/build_dataset.py` rebuilds the analysis table.
- `results/notice_level_scores.csv.gz`: publication number, language, country, stratum, sampling flag, inclusion probability, LLM labels, and method scores (no notice text).
- `logs/annotation_calls.jsonl`, `logs/r1_calls.jsonl`: every LLM request (model, request id, raw output, parsed label, usage/cost). No API key is stored.

## Pipeline
1. `scripts/fetch_ted.py` -> `scripts/build_dataset.py`
2. `scripts/step1_scores_sample.py` (dictionaries M1/M2, multilingual-e5-small and MiniLM embeddings, strata, sample; seed 2027)
3. `scripts/step2_annotate.py` (reference annotation; `scripts/prompts.py` holds the fixed prompt)
4. `scripts/step3_nli.py` (M3, labelled sample only per Amendment 1)
5. `scripts/step4_analysis.py` (confirmatory H1-H4 + pre-specified robustness) -> `results/analysis_main.json`
6. Post hoc: `scripts/step5_exploratory.py`, `scripts/r1_api.py`, `scripts/r1_analyses.py`, `scripts/r2_analyses.py`, `scripts/r3_analyses.py`
- Figures: `figures/make_paper_figures_v2.py` reads `results/*.json`.
- Environment: Python 3.13, sentence-transformers 6.3.0, scikit-learn 1.9.1, statsmodels 0.15.0; CPU only (8 threads on a shared 20-thread machine).

## Licence
Code: MIT. Derived data: CC BY 4.0 with attribution to TED ((c) European Union). The manuscript itself is not part of this repository.

