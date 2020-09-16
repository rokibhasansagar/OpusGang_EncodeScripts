FFVideoSource()
Crop(0, 148, -0, -148)
l=StackHorizontal(FixColumnBrightness(2,16,253,16,255).FixColumnBrightness(1,16,255,16,252).FixColumnBrightnessProtect(1,16,255,16,242,40).FixColumnBrightnessProtect(1,16,255,16,240,100).BalanceBorders(0,0,1,0).SmoothTweak(saturation=1.4).crop(0,0,2,0),FixColumnBrightness(2,16,253,16,255).crop(2,0,1918,0))
t=l.bbmod(1,0,0,0,thresh=6)
r=t.FixColumnBrightness(1917,16,252,16,255).FixColumnBrightness(1918,16,242,16,255).FixColumnBrightnessProtect(1918,16,255,16,240,30).FixColumnBrightnessProtect(1919,16,225,16,255,110).bbmod(0,0,0,2,thresh=12)
f1=r.FixRowBrightness(782,16,250,16,255).FixRowBrightness(783,16,255,16,250)
f2=FixColumnBrightness(2,16,253,16,255).FixColumnBrightness(1,16,215,16,255).FillMargins(1,0,0,0).BalanceBorders(0,0,0,2).FixRowBrightness(782,16,252,16,255).FixRowBrightness(783,16,255,16,252).bbmod(1,0,0,0,thresh=6)
t1=StackVertical(l.FixRowBrightness(2,16,255,16,242).FixRowBrightness(1,16,207,16,255).FillMargins(0,1,0,0).SmoothTweak(saturation=1.7).crop(0,0,0,2),l.FixRowBrightness(2,16,255,16,242).crop(0,2,0,782))
r1=t1.FixColumnBrightness(1917,16,252,16,255).FixColumnBrightness(1918,16,242,16,255).FixColumnBrightnessProtect(1918,16,255,16,240,30).FixColumnBrightnessProtect(1919,16,225,16,255,110).bbmod(0,0,0,2,thresh=12)
f3=r1.FixRowBrightness(782,16,250,16,255).FixRowBrightness(783,16,255,16,250)
f4=StackVertical(t.FixColumnBrightness(1917,16,235,16,255).crop(0,0,-2,0).AddBorders(0,0,2,0).crop(0,0,0,782),t.FixColumnBrightness(1917,16,235,16,255).crop(0,0,-2,0).AddBorders(0,0,2,0).FixRowBrightness(781,16,248,16,255).FixRowBrightness(782,16,255,16,240).FixRowBrightness(783,16,150,16,255).bbmod(0,1,0,0,thresh=20).
\SmoothTweak(saturation=1.5).crop(0,782,0,2))
f5=StackHorizontal(FixColumnBrightness(1,16,220,16,255).FillMargins(1,0,0,0).SmoothTweak(saturation=1.5).crop(0,0,2,0),crop(2,0,1916,0),FixColumnBrightness(1918,16,225,16,255).FillMargins(0,0,1,0).SmoothTweak(saturation=1.3).crop(1918,0,2,0)).FixRowBrightness(782,16,250,16,255).FixRowBrightness(783,16,255,16,250).bbmod(1,0,0,0,thresh=6)
f6=FixColumnBrightness(1,16,255,16,250).FixColumnBrightness(0,16,160,16,255).FixColumnBrightness(1918,16,255,16,245).bbmod(0,0,1,1).FixRowBrightness(782,16,250,16,255).FixRowBrightness(783,16,255,16,250).bbmod(1,0,0,0,thresh=6)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalReader("...fix.txt", "SceneType")
f7=f3kdb(15,65,65,65,0,0)
trim(0,107898)+trim(f7,107899,108035)+trim(108036,113451)+trim(f7,113452,113546)+trim(113547,113980)+trim(f7,113981,114002)+trim(114003,0)

fix.txt

type int
default 0

R 0 372 1
R 373 612 2
R 613 80854 1
R 80855 80905 3
R 80906 122395 1
R 122396 122499 4
R 122500 122547 1
R 122548 122627 4
R 122628 123081 1
R 123082 123328 4
R 123329 136747 1
R 136748 136820 5
R 136821 141065 1
R 141066 141275 6
R 141276 143763 1
