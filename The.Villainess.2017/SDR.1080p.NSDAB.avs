FFVideoSource("")
Crop(0,138,-0,-138)

f1=f3kdb(range=15,y=48,cb=0,cr=0,grainy=32,grainc=0,dynamic_grain=false)
f2=f3kdb(range=15,y=48,cb=48,cr=48,grainy=22,grainc=22,dynamic_grain=true)
f3=gradfun3(thr=0.7,thrC=0.4,radius=16,smode=2)
f4=gradfun3(thr=0.5,thrC=0.2,radius=16,smode=2)
f5=AddBorders(0,2,0,2).Deblock(quant=25, aOffset=0, bOffset=0, mmx=true, isse=true).f3kdb(range=15,y=32,cb=32,cr=32,grainy=64,grainc=64,dynamic_grain=true).Crop(0,2,-0,-2)

ConditionalFilter(f1,last,"SceneType","=","1")
ConditionalFilter(f2,last,"SceneType","=","2")
ConditionalFilter(f3,last,"SceneType","=","3")
ConditionalFilter(f4,last,"SceneType","=","4")
ConditionalFilter(f5,last,"SceneType","=","5")

ConditionalReader("~/zones.txt","SceneType")

#test encodes
#Trim(1744,171697).SelectRangeEvery(every=20000, length=250, offset=500, audio=true)

#GER Blu-ray
