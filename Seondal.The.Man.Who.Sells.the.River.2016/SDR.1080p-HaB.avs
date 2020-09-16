FFVideoSource()
Crop(0, 138, -0, -138)
l=StackHorizontal(FixColumnBrightness(1,16,185,16,235).fillmargins(1,0,0,0).smoothtweak(saturation=2.7).crop(0,0,2,0),FixColumnBrightness(2,16,240,16,235).smoothtweak(saturation=1.4).crop(2,0,2,0),FixColumnBrightness(5,16,237,16,235).FixColumnBrightness(4,16,233,16,235).crop(4,0,1916,0))
r=StackHorizontal(l.FixColumnBrightness(1914,16,240,16,235).FixColumnBrightness(1915,16,232,16,235).crop(0,0,1916,0),l.FixColumnBrightness(1917,16,242,16,235).smoothtweak(saturation=1.3).crop(1916,0,2,0),l.FixColumnBrightness(1918,16,185,16,235).FixColumnBrightness(1919,16,73,16,235).BalanceBorders(0,0,0,1,blur=4).
\smoothtweak(saturation=2.5).crop(1918,0,2,0))
last=r.FixRowBrightness(801,16,233,16,235).FixRowBrightness(802,16,230,16,235).FixRowBrightness(803,16,225,16,235)
b1=l.FixRowBrightness(801,16,233,16,235).FixRowBrightness(802,16,230,16,235).FixRowBrightness(803,16,225,16,235)
f1=r.FixRowBrightness(0,16,168,16,235).FixRowBrightness(803,16,177,16,235).BalanceBorders(1,1,0,0,blur=50)
f2=b1.FixColumnBrightness(1911,16,240,16,235).FixColumnBrightness(1914,16,250,16,235).FixColumnBrightness(1915,16,230,16,235).FixColumnBrightness(1916,16,230,16,235).FixColumnBrightness(1917,16,280,16,235).FixColumnBrightness(1918,16,215,16,235).FixColumnBrightness(1919,16,83,16,235)
f3=b1.FixColumnBrightness(1917,16,240,16,235).FixColumnBrightness(1918,16,185,16,235).FixColumnBrightness(1919,16,72,16,235)
f4=StackHorizontal(b1.crop(0,0,1916,0),b1.FixColumnBrightness(1917,16,240,16,235).smoothtweak(saturation=1.5).crop(1916,0,2,0),b1.FixColumnBrightness(1918,16,185,16,235).FixColumnBrightness(1919,16,72,16,235).BalanceBorders(0,0,0,1,blur=10).smoothtweak(saturation=2.5).crop(1918,0,2,0))
f5=r.FixRowBrightness(802,16,232,16,235).FixRowBrightness(803,16,260,16,235)
f6=FixRowBrightness(6,16,245,16,235).FixRowBrightness(5,16,95,16,235).crop(0,2,0,0).AddBorders(0,0,0,2)
f7=f3kdb(grainc=0,grainy=0,y=65,cr=65,cb=65)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalFilter(last, f7, "SceneType", "<", "7")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 3165 3277 1
R 4779 5120 1
R 5940 5981 1
R 6567 6602 1
R 19222 19265 2
R 21662 21734 1
R 21922 21979 1
R 33739 33971 1
R 57599 57705 1
R 58022 58065 1
R 75861 76237 1
R 98238 98323 1
R 98549 98613 1
R 98769 98852 1
R 100423 100499 1
R 134871 134974 3
R 142665 142733 1
R 145321 145455 4
R 154086 154113 7
R 158853 158915 5
158917 5
R 159649 159682 6
