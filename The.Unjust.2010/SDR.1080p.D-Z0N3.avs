=FFVideoSource().Crop(2, 132, -2, -132) #1px black line + 1px fixable but duplicated line are cropped on both sides
s
FixColumnBrightness(1,16,250,16,255).FixColumnBrightnessProtect(0,16,255,16,238,20).FixColumnBrightness(1911,16,250,16,255).FixColumnBrightness(1913,16,245,16,255).FixColumnBrightness(1914,16,238,16,255).bbmod(0,0,0,2,thresh=3)
StackHorizontal(crop(0,0,1914,0),SmoothTweak(saturation=1.4).crop(1914,0,2,0))
f1=StackVertical(crop(0,0,0,812),crop(0,0,0,814).bbmod(0,2,0,0,thresh=9).crop(0,812,0,2),bbmod(0,2,0,0).FixRowBrightness(815,16,255,16,0).crop(0,814,0,2))
f2=StackVertical(s.crop(0,0,0,812),s.crop(0,0,0,814).bbmod(0,2,0,0,thresh=9).crop(0,812,0,2),s.bbmod(0,2,0,0).FixRowBrightness(815,16,255,16,0).crop(0,814,0,2)).FixColumnBrightness(1911,16,245,16,255).FixColumnBrightness(1912,16,248,16,255).FixColumnBrightness(1913,16,230,16,255).bbmod(0,0,0,2)
f3=StackVertical(s.crop(0,0,0,812),s.crop(0,0,0,814).bbmod(0,2,0,0,thresh=9).crop(0,812,0,2),s.bbmod(0,2,0,0).FixRowBrightness(815,16,255,16,0).crop(0,814,0,2)).FixColumnBrightness(4,16,250,16,255).FixColumnBrightness(1,16,248,16,255).FixColumnBrightness(0,16,255,16,238).FixColumnBrightness(1908,16,253,16,255).
\FixColumnBrightness(1910,16,253,16,255).FixColumnBrightness(1911,16,249,16,255).FixColumnBrightnessProtect(1912,16,255,16,252,20).FixColumnBrightness(1913,16,248,16,255).bbmod(2,0,0,2)
f4=crop(4,0,0,0).AddBorders(0,0,4,0)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalReader("...fix.txt", "SceneType")
f5=f3kdb(15,100,100,100,0,0)
Trim(f5,0,794)+Trim(795,0)

fix.txt

type int
default 0

R 0 295 1
R 296 491 2
R 507 794 3
R 28928 29004 4
