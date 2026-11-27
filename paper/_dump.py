import re
L=open('R2.tex',encoding='utf-8').read().split('\n')
pat=re.compile(r'1\.57|1\.84|177\.2|0\.115|0\.153|0\.86|2\.27|495|429|124\.2|1,812|0\.163|1\.83|0\.38|1\.01|1\.10|6\.0|6\.7|18\.8|1\.40|1\.09|1\.91|0\.57|1\.08|2\.36|1\.61|3\.26|0\.317|0\.337|0\.34\b|0\.36\b|0\.35\b')
for i,l in enumerate(L):
    for m in pat.finditer(l):
        print(i+1, '|', l[max(0,m.start()-100):m.end()+30].replace('\n',' '))
