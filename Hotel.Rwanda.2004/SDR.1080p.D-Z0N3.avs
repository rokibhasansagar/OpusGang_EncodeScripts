s=FFVideoSource().crop(0,132,0,-132).FixColumnBrightness(1919,16,108,16,235)
l=s.FixColumnBrightness(0,16,108,16,235)
b=l.BalanceB(0,0,1,1,thresh=50,blur=20)
b
StackHorizontal(crop(0,0,16,0).SmoothTweak(saturation=1.4,usemt=16).crop(0,0,2,0),crop(2,0,0,0))
f1=s.BalanceB(0,0,0,1,thresh=50,blur=20).crop(4,0,0,0).AddBorders(0,0,4,0)
a=l.BalanceB(0,0,1,1,blur=20)
f2=StackHorizontal(a.crop(0,0,16,0).SmoothTweak(saturation=1.4,usemt=16).crop(0,0,2,0),a.crop(2,0,0,0))
f3=s.BalanceB(0,0,0,1,thresh=50,blur=20).crop(2,0,0,0).AddBorders(0,0,2,0)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalReader("... fix.txt", "SceneType")

fix.txt

type int
default 0

R 130549 130610 1
R 131280 131339 2
R 139546 139610 3
R 150899 150950 3
R 150992 151021 3
R 151065 151107 3
R 151163 151233 3
R 156871 156932 3
R 157145 157179 3
R 157362 157388 3
R 157545 157574 3
