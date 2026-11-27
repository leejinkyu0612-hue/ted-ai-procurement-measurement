import pandas as pd, json
s=pd.read_csv("results/notice_level_scores.csv.gz"); fr=pd.read_parquet("data/frame.parquet",columns=["pub","text"])
s=s.merge(fr,on="pub"); L=s[s.sampled==1]
print(L.groupby("stratum").agg(n=("pub","size"),pos_p=("y_p","sum"),pos_s=("y_s","sum")))
d=L[(L.stratum=="D")&(L.y_p==1)]
for _,r in d.head(12).iterrows(): print("--",r.lang,r.y_s,"|",r.text[:260].replace("\n"," "))
