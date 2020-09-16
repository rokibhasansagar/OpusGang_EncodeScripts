a=FFVideoSource()
b=a.Crop(2, 138, -0, -138)
f1=StackVertical(b.FixRowBrightness(3,16,230,16,235).FixRowBrightness(1,16,95,16,235).smoothtweak(saturation=5).fillmargins(0,1,1,0).BalanceBorders(1,0,0,1).crop(0,0,0,2),b.FixRowBrightness(3,16,230,16,235).FixRowBrightness(800,16,228,16,235).FixColumnBrightness(1917,16,232,16,235).
\fillmargins(0,0,1,0).balanceborders(0,0,0,1).crop(0,2,0,800),b.FixRowBrightness(802,16,218,16,235).FixRowBrightness(803,16,105,16,235).FixColumnBrightness(1917,16,232,16,235).smoothtweak(saturation=1.5).fillmargins(0,0,1,0).BalanceBorders(0,0,0,1).BalanceBorders(0,2,0,0,thresh=5).crop(0,802,0,2))
f2=StackVertical(b.FixRowBrightness(4,16,223,16,235).FixRowBrightness(3,16,135,16,235).crop(0,0,0,4),b.FixRowBrightness(794,16,233,16,235).FixRowBrightness(795,16,230,16,235).FixRowBrightness(797,16,240,16,235).FixRowBrightness(798,16,220,16,235).FixRowBrightness(799,16,142,16,235).FixRowBrightness(800,16,48,16,235).crop(0,4,0,800).
\BalanceBorders(2,0,0,0))
ConditionalFilter(f1, f1, f1, "SceneType", "<", "1")
ConditionalFilter(f1, f2, "SceneType", "<", "2")
ConditionalReader("...fix.txt", "SceneType")
Spline36Resizemod(1280, 536,0,0,-1,0)
f3=f3kdb(grainc=5,grainy=5,dither_algo=3,y=100,cb=100,cr=100)
trim(f3,0,355)+trim(356,630)+trim(f3,631,915)+trim(916,1292)+trim(f3,1293,1312)+trim(1313,0)

fix.txt

type int
default 0

R 0 1034 1
R 1035 1312 2
R 1313 144773 1
