# -*- coding: utf-8 -*-
"""S17 upgraded paper figures (v2: flag-rate labels moved below the data with white boxes, direct legend for pre-specified vs exploratory, M5u marker hollow square; tighter spacing).  Based on S14 (editable). All values are read from EXP_실험/results/*.json (design pi).
fig_prevalence.(png|pdf): design-based prevalence estimates with 95% CIs under alternative references/predictors.
fig_dsl_diag.(png|pdf):  DSL regression diagnostics: SE of beta by control set and auxiliary predictor (log scale), beta printed.
Sized for the LNCS text block (12.2 cm = 4.8 in) so that 7-pt text stays readable."""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
L = lambda f: json.load(open(os.path.join(RES, f), encoding="utf-8"))
main, r1, r2, r3, ex = L("analysis_main.json"), L("r1_analyses.json"), L("r2_analyses.json"), L("r3_analyses.json"), L("exploratory_posthoc.json")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, DARK = "#2f5d8a", "#c26a1b", "#8a96a3", "#222222"


def fig_prevalence():
    pv = main["main"]["prevalence"]
    rows = [  # label, est, lo, hi, post hoc?
        ("HT, primary reference", pv["ht_labels_only"], *pv["ht_ci"], False),
        ("DSL (M5), primary reference", pv["dsl"], *pv["dsl_ci"], False),
        ("DSL, adjudicated reference $X_{adj}$", r1["majority_reference"]["DSL"], *r1["majority_reference"]["DSL_ci"], True),
        ("DSL, agreement reference $X_{\\wedge}$", r1["and_recovered"]["DSL"], *r1["and_recovered"]["DSL_ci"], False),
        ("DSL, union reference $X_{\\vee}$", r1["or_recovered"]["DSL"], *r1["or_recovered"]["DSL_ci"], False),
        ("DSL with M5u", ex["prevalence_dsl_g_M5u"]["est"], *ex["prevalence_dsl_g_M5u"]["ci"], True),
        ("DSL, full-frame cross-fitting", r3["fullframe_crossfit_prevalence"]["dsl"], *r3["fullframe_crossfit_prevalence"]["ci"], True),
        ("DSL, M5 refitted per replicate", pv["dsl"], *r3["refit_bootstrap_dsl_prevalence"]["ci"], True),
    ]
    fig, ax = plt.subplots(figsize=(4.8, 1.45))
    y = np.arange(len(rows))[::-1]
    for yi, (lab, e, lo, hi, ph) in zip(y, rows):
        c = ORANGE if ph else BLUE
        ax.plot([lo * 100, hi * 100], [yi, yi], color=c, lw=1.2, solid_capstyle="butt")
        ax.plot(e * 100, yi, "o", ms=3.6, color=c, mfc="white" if ph else c, mew=1.0)
        ax.text(3.72, yi, f"{e*100:.2f} [{lo*100:.2f}, {hi*100:.2f}]", va="center", ha="left", fontsize=6.3, color=DARK)
    m1, m2 = main["main"]["prevalence"]["naive"]["M1_en_dict"] * 100, main["main"]["prevalence"]["naive"]["M2_ml_dict"] * 100
    ax.axvline(m1, color=GREY, lw=0.8, ls="--"); ax.axvline(m2, color=GREY, lw=0.8, ls=":")
    bb = dict(boxstyle="round,pad=0.15", fc="white", ec="none")
    ax.text(m1 - 0.04, -0.62, f"M1 {m1:.2f}%", ha="right", va="center", fontsize=6.0, color="#555")
    ax.text(m2 + 0.04, -0.62, f"M2 {m2:.2f}%", ha="left", va="center", fontsize=6.0, color="#555")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=BLUE, marker="o", ms=3.6, lw=1.2, label="pre-specified"), Line2D([], [], color=ORANGE, marker="o", ms=3.6, mfc="white", lw=1.2, label="exploratory"),
          Line2D([], [], color=GREY, ls="--", lw=0.8, label="M1 flag rate"), Line2D([], [], color=GREY, ls=":", lw=0.8, label="M2 flag rate")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False, fontsize=6.0, handlelength=1.6, columnspacing=1.0)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-1.15, len(rows) - 0.4)
    ax.set_xlim(0, 3.6); ax.set_xticks(np.arange(0, 3.6, 0.5))
    ax.set_xlabel("Estimated share of AI-related notices (%), 95% CI")
    ax.grid(axis="x", color="#e5e5e5", lw=0.5); ax.set_axisbelow(True)
    ax.text(3.72, len(rows) - 0.55, "estimate [95% CI]", ha="left", va="bottom", fontsize=6.0, color="#555")
    fig.savefig(os.path.join(HERE, "fig_prevalence.png"), dpi=600, bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_prevalence.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_dsl_diag():
    d = r2["dsl_lpm_diagnosis"]; op = r2["open_procedures_only"]["DSL_M5"]
    sets = ["no_controls", "country_only", "full"]
    xlab = ["No controls", "Country FE", "Full controls", "Full controls,\nopen procedures"]
    preds = [("M5", "M5 (pre-specified)", BLUE, "o"), ("M5u", "M5u (post hoc)", ORANGE, "s"), ("M2_indicator", "M2 indicator", GREY, "^")]
    fig, ax = plt.subplots(figsize=(4.8, 1.45))
    off = {"M5": -0.18, "M5u": 0.0, "M2_indicator": 0.18}
    for key, lab, c, mk in preds:
        xs, ses, bs = [], [], []
        for i, s in enumerate(sets):
            v = d[f"{s}|{key}"]; xs.append(i + off[key]); ses.append(v["se_pp"]); bs.append(v["beta_pp"])
        if key == "M5":
            xs.append(3 + off[key]); ses.append(op["se_pp"]); bs.append(op["beta_pp"])
        ax.plot(xs, ses, mk, color=c, ms=4, label=lab, mfc=c if key != "M5u" else "white", mew=1.0)
        for x, se, b in zip(xs, ses, bs):
            ax.text(x + 0.07, se, f"{b:.1f}", fontsize=5.8, va="center", ha="left", color=c)
    ax.set_yscale("log"); ax.set_ylim(3, 4000)
    ax.set_xticks(range(4)); ax.set_xticklabels(xlab); ax.set_xlim(-0.5, 3.6)
    ax.set_ylabel("SE of DSL coefficient (pp, log)")
    sch = r3["schur"]
    for i, s in enumerate(sets):
        ax.text(i, 3.6, f"Schur {sch[s]['schur']:.1f}", ha="center", fontsize=5.9, color="#555")
    ax.grid(axis="y", color="#e5e5e5", lw=0.5, which="major"); ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.3, borderaxespad=0.2)
    fig.savefig(os.path.join(HERE, "fig_dsl_diag.png"), dpi=600, bbox_inches="tight"); fig.savefig(os.path.join(HERE, "fig_dsl_diag.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_prevalence(); fig_dsl_diag(); print("ok")






