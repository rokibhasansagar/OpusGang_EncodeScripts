 FFVideoSource()
f1=crop(0,0,-4,0).AddBorders(4,0,0,0)
f2=FixColumnBrightness(1917,16,243,16,235).FixColumnBrightness(1918,16,248,16,235).FixColumnBrightness(2,16,240,16,235).FixColumnBrightness(1,16,240,16,235).FixRowBrightness(2,16,233,16,235).fillmargins(1,0,1,0).BalanceBorders(2,0,1,1)
f3=FixColumnBrightness(1917,16,243,16,235).FixColumnBrightness(1918,16,248,16,235).FixColumnBrightness(2,16,240,16,235).FixColumnBrightness(1,16,240,16,235).fillmargins(1,0,1,0).BalanceBorders(0,0,1,1)
f4=f3.crop(0,4,0,0).AddBorders(0,0,0,4)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 0 424 1
R 1622 45076 2
R 45077 46607 3
R 46608 111153 2
R 111154 112389 3
R 112390 122115 2
R 122116 123913 4
R 123914 164173 2
