import json; r=json.load(open("results/analysis_main.json")); rb=r["robustness"]
for k in ["label_and","label_or","encoder_minilm"]:
    x=rb[k]; print("==",k)
    print({m:(round(v["f1"],3), round(v["recall"],3)) for m,v in x["metrics"].items()})
    print("H1",round(x["H1"]["diff_recall_noneng_minus_eng"],3),[round(c,3) for c in x["H1"]["ci"]],"H2a",round(x["H2a"]["f1_M5_minus_M1"],3),"H3",round(x["H3"]["dsl_minus_M1"],4),[round(c,4) for c in x["H3"]["ci"]])
    print("prev dsl",round(x["prevalence"]["dsl"],4),[round(c,4) for c in x["prevalence"]["dsl_ci"]],"H4 DSL",round(x["H4"]["DSL_M5"]["beta"],3),round(x["H4"]["DSL_M5"]["se"],3),"lab",round(x["H4"]["labeled_only"]["beta"],3))
print(rb["H2_nonENG_only"]); print(rb["H4b_logit_plugin"])
print(r["timing_s"], r.get("step1_sample",{}).get("timing_s"))
import pandas as pd; fr=pd.read_parquet("data/frame.parquet",columns=["lang"]); print("ENG share frame", (fr.lang=="ENG").mean(), (fr.lang=="ENG").sum())
