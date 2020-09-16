s=FFVideoSource().Crop(2, 20, -0, -20)
h=s.FixRowBrightnessProtect(1038,16,240,16,255,55).FixRowBrightnessProtectb(1038,16,250,16,255,195)
h
StackHorizontal(crop(0,0,1916,0),StackVertical(crop(0,0,0,6),SmoothTweak(saturation=1.2,hue2=4,usemt=4).crop(0,6,0,0)).crop(1916,0,0,0))
FillBorder(0,1,1,1).bbmod(0,0,0,2,thresh=7)
StackVertical(SmoothTweak(saturation=1.5,usemt=4).crop(0,0,0,2),crop(0,2,0,1036),SmoothTweak(saturation=2,usemt=4).crop(0,1038,0,0))
a=StackHorizontal(h.crop(0,0,1916,0).bbmod(0,0,0,2),h.crop(1916,0,0,0))
a1=StackHorizontal(a.crop(0,0,1916,0),StackVertical(a.crop(0,0,0,6),a.SmoothTweak(saturation=1.5,usemt=4).crop(0,6,0,0)).crop(1916,0,0,0)).FixColumnBrightnessProtect(1916,16,255,16,245,50).FillBorder(0,1,1,1).bbmod(0,0,0,2,thresh=7)
f1=StackVertical(a1.SmoothTweak(saturation=1.5,usemt=4).crop(0,0,0,2),a1.crop(0,2,0,1036),a1.SmoothTweak(saturation=2,usemt=4).crop(0,1038,0,0))
f2=StackHorizontal(s.FixRowBrightness(0,16,215,16,255).FixRowBrightnessProtect(0,16,240,16,255,110).crop(0,0,1916,0),s.FixRowBrightness(0,16,215,16,255).FixRowBrightnessProtect(0,16,240,16,255,110).SmoothTweak(saturation=1.2,hue2=4,usemt=4).crop(1916,0,0,0)).FillBorder(0,0,1,0).bbmod(0,0,0,2,thresh=7)
f3=StackHorizontal(s.FixRowBrightness(0,16,225,16,255).crop(0,0,1916,0),s.FixRowBrightness(0,16,225,16,255).SmoothTweak(saturation=1.2,hue2=4,usemt=4).crop(1916,0,0,0)).FillBorder(0,0,1,0).bbmod(0,0,0,2,thresh=7)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalReader("...fix.txt", "SceneType")
Spline36ResizeMod(1280, 694)

fix.txt

type int
default 0

R 3027 3150 1
R 57932 57997 1
R 58297 58323 1
R 116272 116907 2
R 116908 117548 3
