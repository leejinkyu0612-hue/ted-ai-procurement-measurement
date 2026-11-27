import json; r=json.load(open("results/analysis_main.json"))
m=r["main"]
for k,v in m["metrics"].items(): print(k, {a:(round(b,3) if isinstance(b,float) else b) for a,b in v.items() if not a.endswith("ci")}, "f1ci", [round(x,3) for x in v["f1_ci"]])
print("auprc", m["auprc"]); print("H1",m["H1"]); print("H2a",m["H2a"]); print("H2b",m["H2b"]); print("H3", m["H3"])
print("strata", r["strata"], "missing", r["missing_primary_labels_in_sample"])
print("thr", r["robustness"]["thresholds"])
print("expl", json.dumps(r["exploratory"]["recall_by_language"])[:800])
