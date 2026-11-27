import json,collections
rows=[json.loads(l) for l in open('logs/r1_calls.jsonl',encoding='utf-8')]
print('requests',len(rows))
for t in ['adjudicate','reparse_secondary']:
    r=[x for x in rows if x['task']==t]; pubs=set(x['pub'] for x in r); ok=set(x['pub'] for x in r if x.get('label') is not None)
    print(t,'requests',len(r),'notices',len(pubs),'parsed',len(ok))
print('cost',round(sum((x.get('usage') or {}).get('cost',0) or 0 for x in rows),4))
print(rows[0]['ts'], rows[-1]['ts'])
