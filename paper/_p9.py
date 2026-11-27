import re
s=open('Lee_Sung_AIBiz2027_final.tex',encoding='utf-8').read()
new=open('s20_abstract.txt',encoding='utf-8').read().strip()
new_body=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',new,re.S).group(1).strip()
m=re.search(r'(\\begin\{abstract\}\s*)(.*?)(\s*\\keywords)',s,re.S); old_body=m.group(2).strip()
def nums(t): return sorted(re.findall(r'\d+(?:[.,]\d+)*',t))
def words(t): return len(re.findall(r'[A-Za-z0-9]+',re.sub(r'\\[a-zA-Z]+','',t)))
print('old words',words(old_body),'new words',words(new_body)); print('numbers same',nums(old_body)==nums(new_body)); print(set(nums(old_body))^set(nums(new_body)))
ok = nums(old_body)==nums(new_body) and words(new_body)<=words(old_body)
if ok:
    s=s[:m.start(2)]+new_body+s[m.end(2):]
old_t=re.search(r'\\title\{([^}]*)\}',s).group(1)
new_t='Identifying AI-Related IT and Software Award Notices in Tenders Electronic Daily: Keyword Dictionaries, Embedding Classifiers, and Design-Based Supervised Learning'
s=s.replace('\\title{'+old_t+'}','\\title{'+new_t+'}')
s=re.sub(r'\\titlerunning\{[^}]*\}', r'\\titlerunning{Identifying AI-Related Award Notices in TED}', s)
open('Lee_Sung_AIBiz2027_final.tex','w',encoding='utf-8').write(s)
open('s20_applied.txt','w',encoding='utf-8').write(f"abstract applied: {ok}\nold title: {old_t}\nnew title: {new_t}\nold words {words(old_body)} new words {words(new_body)}\n")
print('applied', ok)
