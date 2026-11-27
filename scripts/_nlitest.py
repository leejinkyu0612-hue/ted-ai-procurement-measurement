import time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.set_num_threads(4)
n="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"; tok=AutoTokenizer.from_pretrained(n); m=AutoModelForSequenceClassification.from_pretrained(n).eval()
fr=pd.read_parquet("data/sample_recon.parquet"); tx=fr.text.tolist()[:16]
t=time.time()
with torch.inference_mode():
    enc=tok(tx,["This procurement is about artificial intelligence."]*16,truncation="only_first",max_length=256,padding=True,return_tensors="pt")
    print(enc["input_ids"].shape); m(**enc)
print("16 in", time.time()-t, "s")
