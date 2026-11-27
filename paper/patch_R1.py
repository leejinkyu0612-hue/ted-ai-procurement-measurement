# -*- coding: utf-8 -*-
"""Apply author/fact fixes to Claude's R1 revision (claude_R1_revised.tex) -> R1.tex. Every replacement must match once."""
import re, sys, json
src = open("claude_R1_revised.tex", encoding="utf-8").read()
r1 = json.load(open("../results/r1_analyses.json", encoding="utf-8"))
A, O = r1["and_recovered"], r1["or_recovered"]
reps = [
 (r"""% TODO: authors and affiliations (single-blind review: names may appear).
\author{First Author\inst{1} \and Second Author\inst{1}}
\authorrunning{F. Author and S. Author}
\institute{Institution, City, Country\\\email{author@example.org}}""",
  r"""\author{Jinkyu Lee\orcidID{0000-0002-0886-1572} \and Chang Soo Sung\thanks{Corresponding author.}\orcidID{0000-0001-9284-817X}}
\authorrunning{J. Lee and C. S. Sung}
\institute{Department of Technology Entrepreneurship, Graduate School, Dongguk University, Seoul, Republic of Korea\\ \email{leejinkyu0612@naver.com, redsun44@dgu.ac.kr}}"""),
 ("Code, prompts, and notice-level derived data are released % TODO: repository URL and commit hash", "Code, prompts, and notice-level derived data are released (see Data and code availability)"),
 ("% TODO: state whether the third model labelled all 1,199 notices or only disagreements, and how ties are resolved for the 33 notices without a secondary label.",
  "The third model labelled only the 123 notices where the first two disagreed or the secondary label was missing, and all primary positives in strata C and D; where two labels were available and tied, its label decided."),
 ("% TODO: confirm (a) E5 'query:'/'passage:' prefixes, pooling and normalisation; (b) TF-IDF vocabulary fitted within training folds; (c) how predictions for unlabelled notices are formed (average of fold models or refit on all labels). Implementation details belong in the repository.",
  "E5 inputs use the `passage:'/`query:' prefixes with mean pooling and $L_2$ normalisation; TF-IDF vocabularies are fitted within training folds, and unlabelled notices are scored by a model refitted on all labels."),
 ("After deduplication, % TODO: state the rule (publication number; handling of corrigenda/re-publications) and main vs. additional CPV.", "After deduplication by publication number (CPV matching includes additional CPV codes),"),
 ("% TODO: give the eForms business-term identifier of the tender-count field and the submission type used (all tenders vs. SME/electronic) in the repository.",
  "The count is the eForms received-submissions value of type `tenders' (all tenders, not the SME or electronic sub-counts)."),
 ("CPV division, % TODO: report the number of CPV-division levels actually used.", f"CPV division ({r1['n_cpv_levels_used']} levels after pooling divisions with fewer than 30 notices),"),
 ("% TODO: time and cost of the re-query and of the third-model adjudication.", "The re-query and the adjudication (209 requests in total) cost US\\$0.03."),
 ("% TODO: state whether M5 and thresholds are refitted in each bootstrap replicate.", "classifiers and thresholds are not refitted within replicates, so these CIs reflect label sampling only."),
 ("committed to the project repository % TODO: commit hash and date; amendment commit.", "committed to the project repository (commit 7314e8d, 27 November 2026, 11:12 KST; amendment commit b1eb797, 12:22 KST)"),
 ("% TODO: confirm that logged recall_eng=0.076 / recall_noneng=0.351 refer to the fold-internal threshold (not 0.5).\n", ""),
]
for a, b in reps:
    n = src.count(a)
    if n != 1: sys.exit(f"MATCH {n}: {a[:70]}")
    src = src.replace(a, b)
# sensitivity table: replace pre-requery X_and row with recovered X_and and X_or rows
old_row = r"$X_\wedge$$^\star$ & -- & -- & -- & 1.41 [1.10, 1.91] & -- \\"
assert src.count(old_row) == 1
f = lambda x: f"{x:.2f}"
new_rows = (f"$X_\\wedge$ & {A['n_pos']} & {f(A['M1']['recall'])} & {f(A['M2']['f1'])} [{f(A['M2']['f1_ci'][0])}, {f(A['M2']['f1_ci'][1])}] & {f(A['DSL']*100)} [{f(A['DSL_ci'][0]*100)}, {f(A['DSL_ci'][1]*100)}] & {f(A['H3_diff']*100)} [{f(A['H3_ci'][0]*100)}, {f(A['H3_ci'][1]*100)}] \\\\\n"
            f"$X_\\vee$ & {O['n_pos']} & {f(O['M1']['recall'])} & {f(O['M2']['f1'])} [{f(O['M2']['f1_ci'][0])}, {f(O['M2']['f1_ci'][1])}] & {f(O['DSL']*100)} [{f(O['DSL_ci'][0]*100)}, {f(O['DSL_ci'][1]*100)}] & {f(O['H3_diff']*100)} [{f(O['H3_ci'][0]*100)}, {f(O['H3_ci'][1]*100)}] \\\\")
src = src.replace(old_row, new_rows)
src = src.replace(r"$^\star$Initial secondary labels.", "$X_\\wedge$/$X_\\vee$ use the recovered secondary labels (missing secondary: primary label).")
old_txt = re.search(r"Under \$X_\\wedge\$, computed before the re-query.*?results matched the primary analysis\.", src, re.S).group(0)
new_txt = (f"With the recovered secondary labels, the agreement reference $X_\\wedge$ kept one stratum-D positive; it gave a DSL prevalence of {f(A['DSL']*100)}\\% (CI {f(A['DSL_ci'][0]*100)}--{f(A['DSL_ci'][1]*100)}\\%), the highest M2 F1 ({f(A['M2']['f1'])}), and a lower recall of M1 for non-English than for English notices ($\\Delta_{{\\mathrm{{H1}}}}={A['H1_diff']:.2f}$, CI {A['H1_ci'][0]:.2f} to {A['H1_ci'][1]:.2f}); the union reference $X_\\vee$ matched the primary analysis.")
src = src.replace(old_txt, new_txt)
# H4 row for X_and: recovered labels
old_h4 = re.search(r"DSL with M5, agreement reference \(\$X_\\wedge\$\)\$\^\\star\$ & -- & 5\.7 & \[\$-7\.0\$, 18\.4\] \\\\", src)
if old_h4:
    b, s = A["DSL_beta"] * 100, A["DSL_se"] * 100
    src = src.replace(old_h4.group(0), f"DSL with M5, agreement reference ($X_\\wedge$) & -- & {b:.1f} & [${b-1.96*s:.1f}$, {b+1.96*s:.1f}] \\\\")
else:
    sys.exit("H4 and-row not found")
# credits block -> ours
cb = re.search(r"\\begin\{credits\}.*?\\end\{credits\}", src, re.S).group(0)
src = src.replace(cb, open("sec_back.tex", encoding="utf-8").read().strip())
src = src.replace(r"\bibliography{refs_placeholder}", r"\bibliography{refs}")
open("R1.tex", "w", encoding="utf-8").write(src)
print("ok; remaining TODO:", src.count("% TODO"), "tbd:", src.count("\\tbd"))
