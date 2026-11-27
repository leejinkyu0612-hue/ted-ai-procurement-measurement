import json,collections
c=collections.Counter(); cost=0
for l in open('logs/r1_calls.jsonl',encoding='utf-8'):
    j=json.loads(l); c[(j['task'],j.get('label'))]+=1; cost+=(j.get('usage') or {}).get('cost',0) or 0
print(dict(c), round(cost,4))
