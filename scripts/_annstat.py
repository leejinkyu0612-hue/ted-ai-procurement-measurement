import json,collections
c=collections.Counter(); cost=0; err=0
for l in open('logs/annotation_calls.jsonl',encoding='utf-8'):
    j=json.loads(l)
    if j.get('label') is None: err+=1
    else: c[(j['model'][:12],j['label'])]+=1
    cost+=(j.get('usage') or {}).get('cost',0) or 0
print(dict(c),'err',err,'cost',round(cost,4))
