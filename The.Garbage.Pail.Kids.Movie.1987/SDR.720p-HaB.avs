FFVideoSource()
Crop(0, 20, -0, -20)
StackHorizontal(FixColumnBrightness(1,16,180,16,235).FixColumnBrightness(0,16,75,16,235).smoothtweak(saturation=3.2).BalanceBorders(0,0,1,0,blur=15).crop(0,0,2,0),FixColumnBrightness(1914,16,225,16,235).FixColumnBrightness(1915,16,250,16,235).crop(2,0,1914,0),
\FixColumnBrightness(1916,16,225,16,235).FixColumnBrightness(1917,16,140,16,235).crop(0,0,1918,0).BalanceBorders(0,0,0,2,thresh=16,blur=20).crop(1916,0,2,0),FixColumnBrightness(1918,16,103,16,235).FixColumnBrightness(1919,16,100,16,235).smoothtweak(saturation=2.8).BalanceBorders(0,0,0,1,blur=5).crop(1918,0,2,0))
FixRowBrightness(1037,16,230,16,235)
Spline36Resizemod(1280, 694)
