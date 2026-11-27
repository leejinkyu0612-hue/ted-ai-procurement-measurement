from PIL import Image; import glob
ims=[Image.open(f) for f in sorted(glob.glob('_pg-*.png'))]
w=sum(i.width for i in ims); h=max(i.height for i in ims); m=Image.new('RGB',(w,h),'white'); x=0
for i in ims: m.paste(i,(x,0)); x+=i.width
m.save('_montage.png'); print(m.size)
