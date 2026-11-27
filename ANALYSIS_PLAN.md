# Pre-specified analysis plan (사전계획) — P1
Fixed on 2026-11-27 before any annotation, classifier training, or outcome analysis. Committed to a private GitHub repository before experiments. Any analysis not listed here is reported as exploratory/post hoc. Changes are recorded as dated amendments (AMENDMENT_n.md) before the affected results are computed.

Status at fixing time: raw TED data downloaded (55,508 unique notices) and profiled for counts/missingness only (results/dataset_profile.json). No AI label, classifier score, or competition outcome has been computed. One format test call per annotation model on a synthetic German sentence (not a TED notice) was made to check output parsing (cost < US$0.0001).

## 1. Research questions
- RQ1 (measurement): How accurately do six scalable methods identify AI-related procurement notices relative to a codebook-based reference annotation, overall and by notice language?
- RQ2 (prevalence): How much do prevalence estimates differ across methods, and relative to a design-based (DSL) estimate?
- RQ3 (downstream): Is AI-relatedness associated with single bidding, and does the estimated association depend on the measurement method?

## 2. Data
- Source: TED Search API v3, notice-type = can-standard, classification-cpv = 72000000 or 48000000 (hierarchical match), publication dates 2024-01-01 to 2025-12-31 (scripts/fetch_ted.py). Reuse: TED legal notice (free reuse, commercial or non-commercial; Commission Decision 2011/833/EU).
- Unit: notice (publication-number), deduplicated. Text = procedure title + ". " + procedure description (fallback lot title/description) in the first official language; truncated to 2,000 characters (1,500 for LLM annotation).
- Frame F: notices with title+description length ≥ 10 characters.
- Language group: ENG vs non-ENG (first official language).

## 3. Construct and codebook (reference annotation)
AI-related = the procured object includes the development, supply, licensing, operation, or consulting/training on a system or service whose core functionality relies on AI techniques: machine learning, deep learning, natural language processing or large language models, chatbots/conversational agents, computer vision/image recognition, speech recognition/synthesis, or predictive models learned from data, including explicitly AI-described automation. Not AI-related: generic software, ERP, websites, RPA/workflow automation without AI, BI dashboards/data warehouses without learning components, "smart" devices without AI functionality, hardware only.
- Prompt (fixed): see scripts/prompts.py (ANNOTATION_PROMPT). Output JSON {"ai": 0|1}.
- Primary reference annotator: anthropic/claude-haiku-5.5 via OpenRouter (reasoning disabled, temperature 0, max_tokens 30). Secondary annotator: google/gemini-2.5-flash-lite (temperature 0, max_tokens 30). Parse failures retried twice; remaining failures reported as missing and excluded from the labeled set (their sampling probability is unchanged; this is reported as a deviation risk).
- Reference label X = primary annotator. Agreement: Cohen's kappa (weighted by 1/π), overall and by language group. Sensitivity definitions: X_and (both = 1), X_or (either = 1).
- The reference is an LLM codebook annotation, not expert ground truth; this limits construct validity and is stated as a limitation.

## 4. Measurement methods (applied to all of F unless noted)
- M1 English dictionary: case-insensitive regex for English AI terms (scripts/dictionaries.py, EN list), applied to all notices regardless of language.
- M2 Multilingual dictionary: M1 terms + core terms for the 24 EU languages ("artificial intelligence", "machine learning", "neural network", "language model", "chatbot" equivalents) + whole-word uppercase abbreviations KI and IA.
- M3 Zero-shot NLI: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli, hypothesis "This procurement is about artificial intelligence."; entailment probability ≥ 0.5 (sensitivity 0.3, 0.7). If CPU time for F exceeds 6 h, M3 is evaluated on the labeled sample only (prevalence/downstream for M3 then not reported).
- M4 Embedding prototype similarity: intfloat/multilingual-e5-small, max cosine similarity between "passage: " text and five fixed English "query: " prototypes (scripts/prompts.py); threshold chosen within training folds to maximize weighted F1 (same folds as M5).
- M5 Supervised embedding classifier (proposed measurement): logistic regression (L2, C = 1.0, sample_weight = 1/π) on multilingual-e5-small embeddings trained on the reference labels; probability ≥ 0.5 (sensitivity 0.3, 0.7).
- M6 Supervised TF-IDF baseline: character 3–5-gram TF-IDF (max 200,000 features) + same logistic regression.
- Robustness encoder for M4/M5: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2.
- Evaluation of M4–M6 on the labeled sample uses 5-fold cross-fitted predictions (folds stratified by sampling stratum, seed 2027). Predictions for unlabeled notices use a model trained on all labeled notices.

## 5. Reference sample design
- Strata (computed on F): A = M1 hit; B = M2 hit and not M1; C = neither, and M4 similarity in the top 3% of F; D = remaining notices.
- Sample sizes (simple random sampling without replacement within strata, seed 2027): A min(250, N_A); B min(250, N_B); C 250; D 450. π_i = n_h / N_h.

## 6. Hypotheses and tests (confirmatory)
- H1: For M1, weighted recall is lower for non-English than for English notices (supported if the upper bound of the 95% stratified bootstrap CI of recall_nonENG − recall_ENG < 0).
- H2: M5 has higher weighted F1 than M1 and than M2 (paired stratified bootstrap; supported if lower 95% CI bound of each difference > 0).
- H3: The DSL prevalence estimate exceeds the M1 naive prevalence (supported if lower 95% CI bound of DSL − M1 > 0).
- Multiplicity: Holm correction over the four bootstrap p-values (H1, H2a, H2b, H3).
- H4 (two-sided): In the regression sample, the DSL estimate of the coefficient on AI-relatedness in a linear probability model of single bidding differs from zero (95% sandwich CI). H4b (descriptive): naive plug-in coefficients for M1–M6 are reported next to the DSL estimate and the labeled-only weighted estimate.
- Bootstrap: 2,000 replicates, resampling within strata of the labeled sample (models are not refitted inside the bootstrap; cross-fitted predictions are fixed).

## 7. Downstream regression
- Regression sample: F ∩ exactly one "tenders" field ∩ 1 ≤ tenders ≤ 100.
- Outcome: single_bid = 1[tenders = 1].
- Covariates: AI-relatedness (per method), fixed effects for buyer country, CPV division (divisions with < 30 notices pooled), procedure type, buyer legal type (missing as own category), publication quarter; log(1 + total value) with a missing-value indicator (currency scale absorbed by country fixed effects).
- DSL: X̃ = ĝ + (R/π)(X − ĝ) with ĝ = M5 probability (cross-fitted for labeled notices); moment conditions of OLS with X and X² replaced by X̃ (valid because X is binary); sandwich standard errors.
- Interpretation: association only; no causal claim.

## 8. Robustness (pre-specified)
Reference label definitions X_and and X_or; robustness encoder; thresholds 0.3/0.5/0.7; excluding English notices for H2; logistic regression (plug-in) instead of LPM for H4b.

## 9. Efficiency
CPU wall-clock time per 1,000 notices for each method (8 threads, below-normal priority, shared machine; reported as indicative).

## 10. Exploratory (labeled as such in the paper)
Per-language recall; country ranking stability; annotator agreement by language; tender-count (Poisson) models; SME winners; larger encoder (multilingual-e5-large) if time allows.
