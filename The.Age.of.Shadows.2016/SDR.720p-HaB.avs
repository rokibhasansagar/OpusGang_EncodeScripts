FFVideoSource()
Crop(0, 138, -0, -138)
f1=FixColumnBrightness(1919,16,225,16,235).FixColumnBrightness(0,16,232,16,235).BalanceBorders(0,0,1,1)
f2=FixColumnBrightness(1919,16,175,16,235).FixColumnBrightness(0,16,232,16,235).BalanceBorders(0,0,0,1)
f3=fillmargins(1,0,1,0).BalanceBorders(0,0,1,1)
f4=StackHorizontal(crop(0,0,1918,0),chromashift(c=2).smoothtweak(hue1=-2).crop(1918,0,2,0)).FixColumnBrightness(1,16,210,16,235).FixColumnBrightness(1918,16,240,16,235).fillmargins(1,0,1,0).BalanceBorders(0,0,1,1)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalReader("...fix.txt", "SceneType")
Spline36Resizemod(1280, 536)

fix.txt

type int
default 0

R 0 65842 1
R 65843 65978 2
R 65979 66221 1
R 66222 66320 2
R 66321 134169 1
R 134170 134268 3
134269 1
R 134270 134272 3
R 134273 141381 1
R 141449 141462 4
R 141463 195697 1
