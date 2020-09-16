 FFVideoSource()
f1=fillmargins(1,0,1,0).BalanceBorders(0,0,1,1)
f2=FixColumnBrightness(3,16,240,16,235).FixColumnBrightness(2,16,250,16,235).FixColumnBrightness(0,16,205,16,235)
f3=FixRowBrightness(2,16,225,16,235).FixRowBrightness(1,16,255,16,235).FixColumnBrightness(2,16,225,16,235).FixColumnBrightness(1,16,250,16,235).FixColumnBrightness(1916,16,230,16,235).FixColumnBrightness(1917,16,232,16,235).FixColumnBrightness(1918,16,260,16,235).FixColumnBrightness(1919,16,158,16,235).FixRowBrightness(1076,16,231,16,235).FixRowBrightness(1077,16,233,16,235).FixRowBrightness(1078,16,260,16,235).FixRowBrightness(1079,16,160,16,235)
f4=FixRowBrightness(0,16,198,16,235).FixColumnBrightness(1919,16,198,16,235).FixColumnBrightness(0,16,198,16,235).FixRowBrightness(1079,16,198,16,235)
f5=FixColumnBrightness(2,16,230,16,235).FixColumnBrightness(1,16,250,16,235).FixColumnBrightness(1918,16,250,16,235).FixColumnBrightness(1919,16,165,16,235).FixRowBrightness(1076,16,232,16,235).FixRowBrightness(1077,16,233,16,235).FixRowBrightness(1078,16,255,16,235).FixRowBrightness(1079,16,170,16,235).FixRowBrightness(3,16,230,16,235).FixRowBrightness(2,16,223,16,235).FixRowBrightness(1,16,245,16,235).FixRowBrightness(0,16,230,16,235)
f6=fillmargins(0,0,1,0).BalanceBorders(0,0,0,1)
f7=FixRowBrightness(1078,16,255,16,235).FixRowBrightness(1079,16,165,16,235).FixColumnBrightness(1915,16,232,16,235).FixColumnBrightness(1916,16,236,16,235).FixColumnBrightness(1917,16,260,16,235).FixColumnBrightness(1919,16,125,16,235).FixRowBrightness(2,16,225,16,235).FixRowBrightness(1,16,248,16,235).FixColumnBrightness(3,16,232,16,235).FixColumnBrightness(1,16,255,16,235).FixColumnBrightness(0,16,230,16,235)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalFilter(last, f7, "SceneType", "<", "7")
ConditionalReader("...fix.txt", "SceneType")
Spline36Resizemod(1280, 720)

fix.txt

type int
default 0

R 12089 13080 1
R 13208 13403 1
R 29277 29318 2
R 39718 39786 3
R 40395 40765 4
R 43903 44396 5
R 98614 98676 6
R 98829 98878 6
R 107142 107442 7
