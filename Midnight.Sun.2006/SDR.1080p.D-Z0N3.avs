FFVideoSource()
Crop(0, 24, -0, -22)
f=StackVertical(FixRowBrightness(0,16,174,16,255).FixRowBrightness(1031,16,253,16,255).crop(0,0,0,1032),FixRowBrightness(1032,16,173,16,255).FixRowBrightness(1033,16,80,16,255).Balanceborders(0,1,0,0,blur=10).crop(0,1032,0,2))
StackHorizontal(f.FixColumnBrightness(1,16,150,16,255).FillMargins(1,0,0,0).SmoothTweak(saturation=2).crop(0,0,2,0),f.FixColumnBrightness(2,16,205,16,255).SmoothTweak(saturation=1.3).crop(2,0,2,0),StackVertical(FixRowBrightness(0,16,174,16,255).FixRowBrightness(1031,16,253,16,255).crop(0,0,0,1032),FixRowBrightness(1032,16,173,16,255)
\.FixRowBrightness(1033,16,80,16,255).Balanceborders(0,1,0,0,blur=10).SmoothTweak(saturation=2.3).crop(0,1032,0,2)).crop(4,0,1912,0),f.FixColumnBrightness(1917,16,205,16,255).SmoothTweak(saturation=1.3).crop(1916,0,2,0),f.FixColumnBrightness(1918,16,150,16,255).FixColumnBrightness(1919,16,90,16,255).SmoothTweak(saturation=2.2).
\crop(1918,0,2,0))
