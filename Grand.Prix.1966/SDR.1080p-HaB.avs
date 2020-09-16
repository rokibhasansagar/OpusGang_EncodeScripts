FFVideoSource()
Crop(0, 104, -0, -104)
StackVertical(FixColumnBrightness(2,16,240,16,235).FixColumnBrightness(1,16,255,16,235).FixColumnBrightness(0,16,115,16,235).FixColumnBrightness(1916,16,234,16,235).FixColumnBrightness(1917,16,240,16,235).FixColumnBrightness(1918,16,260,16,235).fillmargins(0,1,0,1).BalanceBorders(1,0,1,1).
\smoothtweak(saturation=1.9).crop(0,0,0,2),StackHorizontal(FixColumnBrightness(1,16,255,16,235).FixColumnBrightness(0,16,115,16,235).fillmargins(0,0,0,1).BalanceBorders(0,1,1,0).smoothtweak(saturation=1.6).crop(0,0,2,0),FixColumnBrightness(2,16,240,16,235).FixColumnBrightness(1916,16,234,16,235).
\FixColumnBrightness(1917,16,240,16,235).FixColumnBrightness(1918,16,260,16,235).fillmargins(0,0,1,1).BalanceBorders(0,1,0,1).crop(2,0,1918,0)).crop(0,2,0,870))
StackVertical(crop(0,0,0,870),smoothtweak(saturation=2).crop(0,870,0,0))
