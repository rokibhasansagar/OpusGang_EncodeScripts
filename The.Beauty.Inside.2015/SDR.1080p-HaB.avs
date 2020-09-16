FFVideoSource()
Crop(0, 138, -0, -138)
FixColumnBrightness(0,16,217,16,235)
FixColumnBrightness(1919,16,223,16,235)
f1=FixColumnBrightness(2,16,230,16,235).FixColumnBrightness(1916,16,230,16,235).FixColumnBrightness(1917,16,225,16,235).BalanceBorders(0,0,2,2)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 136367 136416 1
