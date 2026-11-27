import json
r=json.load(open("results/analysis_main.json")); m=r["main"]; L=[]
L.append(f"[Frame] N={m['prevalence']['N_frame']}; strata {r['strata']}; missing primary labels {r['missing_primary_labels_in_sample']}; step1 {r.get('step1_sample',{}).get('timing_s')}")
for k,v in m["metrics"].items(): L.append(f"[Accuracy {k}] P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} CI {v['f1_ci'][0]:.3f}-{v['f1_ci'][1]:.3f} R_EN={v['recall_eng']:.3f} R_non={v['recall_noneng']:.3f}")
L.append(f"[AUPRC] {m['auprc']}")
L.append(f"[H1] {m['H1']} [H2a] {m['H2a']} [H2b] {m['H2b']} [H3] {m['H3']} [Holm] {m['holm_adjusted']}")
L.append(f"[Prevalence] {m['prevalence']}")
L.append(f"[H4] " + json.dumps(m['H4']))
L.append(f"[Agreement (initial)] {r['agreement']}")
L.append(f"[Robustness H2 non-ENG F1] {r['robustness']['H2_nonENG_only']}; [logit plug-in] {r['robustness']['H4b_logit_plugin']}; [MiniLM M5 F1] {r['robustness']['encoder_minilm']['metrics']['M5_emb_lr']['f1']:.3f}")
e=json.load(open("results/exploratory_posthoc.json")); L.append("[Exploratory M5u] " + json.dumps({k:{kk:round(vv,3) if isinstance(vv,float) else vv for kk,vv in v.items() if kk in ('precision','recall','f1','recall_eng','recall_noneng')} for k,v in e['metrics'].items()}) + f" AUPRC {e['auprc']} M3 AUPRC {e.get('auprc_M3')}")
L.append("[Annotation] 2,400 primary+secondary requests 11:54-12:16 KST, cost US$0.138; R1 re-query+adjudication 275 requests US$0.0316; Haiku primary positives 364/1,199; secondary parsed 1,114 initially, 53 of 86 recovered")
open("results/main_digest.txt","w",encoding="utf-8").write("\n".join(L)); print(len("\n".join(L)))
