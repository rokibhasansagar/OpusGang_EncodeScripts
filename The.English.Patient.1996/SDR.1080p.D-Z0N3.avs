a=FFVideoSource().Crop(0, 20, -0, -20)
l=StackHorizontal(a.FixColumnBrightness(0,16,125,16,255).Smoothtweak(saturation=2.5).crop(0,0,2,0),a.crop(2,0,0,0))
b=l.FixRowBrightnessProtect(1037,16,245,16,255,200).FixRowBrightnessProtect(1038,16,220,16,255,200).FixRowBrightnessProtect(1039,16,230,16,255,200)
tb=StackVertical(b.SmoothTweak(saturation=1.2).crop(0,0,0,2),b.crop(0,2,0,1036),b.SmoothTweak(saturation=1.2).crop(0,1038,0,0))
r=tb.FixColumnBrightness(1919,16,125,16,255)
r
BalanceBorders(0,0,1,1,thresh=50,blur=110)
fa=a.FixRowBrightnessProtect(1034,16,240,16,255,50).FixRowBrightnessProtect(1035,16,255,16,235,90).FixRowBrightness(1036,16,240,16,255)
f1=StackVertical(fa.crop(0,0,0,1036),SmoothTweak(saturation=1.7).crop(0,1036,0,2),fa.crop(0,1038,0,0))
f2=tb.BalanceBorders(0,0,1,0,thresh=50,blur=110).bbmod(0,0,0,2)
f3=tb.BalanceBorders(0,0,1,0,thresh=50,blur=110).FixColumnBrightness(1917,16,235,16,255).bbmod(0,0,0,2)
f4=tb.FixColumnBrightness(1914,16,250,16,255).FixColumnBrightness(1915,16,240,16,255).FixColumnBrightness(1916,16,230,16,255).FixColumnBrightness(1917,16,180,16,255).FixColumnBrightness(1918,16,200,16,255).FixColumnBrightness(1919,16,100,16,255).BalanceBorders(0,0,1,0,thresh=50,blur=110).BalanceBorders(0,0,0,2)
f5=tb.FixColumnBrightness(1914,16,250,16,255).FixColumnBrightness(1915,16,240,16,255).FixColumnBrightness(1916,16,230,16,255).FixColumnBrightness(1917,16,210,16,255).FixColumnBrightness(1918,16,200,16,255).FixColumnBrightness(1919,16,100,16,255).BalanceBorders(0,0,1,0,thresh=50,blur=110).BalanceBorders(0,0,0,2)
f6=tb.BalanceBorders(0,0,1,0,thresh=50,blur=110).crop(0,0,-14,0).AddBorders(14,0,0,0)
f7=tb.BalanceBorders(0,0,1,0,thresh=50,blur=110).crop(0,0,-2,0).AddBorders(2,0,0,0)
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

R 52 225 1
R 22137 22224 2
R 22225 22227 3
22228 2
22229 3
22230 2
22231 3
22232 2
22233 3
R 22234 22240 2
22241 3
169169 4
R 169170 169215 5
R 169476 169523 6
R 172061 172139 7
