FFVideoSource()
Crop(0, 138, -0, -138)
FixRowBrightness(2,16,235,16,235).FixRowBrightness(1,16,260,16,235).FixRowBrightness(0,16,255,16,235).FixRowBrightness(801,16,233,16,235).FixRowBrightness(802,16,248,16,235).FixRowBrightness(803,16,235,16,235).BalanceBorders(0,2,0,0,thresh=16).BalanceBorders(1,0,0,0)
StackVertical(crop(0,0,0,2),crop(0,2,0,802).BalanceBorders(1,0,0,0,thresh=16))
f1=FixColumnBrightness(2,16,252,16,235).FixColumnBrightness(1,16,245,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,3,0)
f2=FixColumnBrightness(1915,16,233,16,235).FixColumnBrightness(1916,16,230,16,235).FixColumnBrightness(1918,16,247,16,235).FixColumnBrightness(1919,16,238,16,235).BalanceBorders(0,0,0,3,thresh=25)
f3=FixColumnBrightness(1918,16,230,16,235).BalanceBorders(0,0,3,1)
f4=BalanceBorders(0,0,0,3,thresh=25)
f5=f3kdb(grainc=0,grainy=0,dither_algo=3,y=150)
f6=f3kdb(grainc=0,grainy=0,dither_algo=3,y=100)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 135 393 5
R 3778 4034 1
R 10967 11176 6
R 68177 68572 2
R 70382 70947 2
R 77310 77500 3
R 79801 80634 4
R 86660 86729 4
R 87268 87506 4
R 87673 87748 4
R 87812 87892 4
R 88235 88905 4
R 120104 120686 4
