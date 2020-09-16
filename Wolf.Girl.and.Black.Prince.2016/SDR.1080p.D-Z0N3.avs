FFVideoSource()
Crop(0, 20, -0, -20)
s=last
StackVertical(crop(0,0,0,6).FillBorder(0,1,0,0).crop(0,0,0,2).Tweak(sat=2,dither=true),crop(0,2,0,1036),crop(0,1034,0,0).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=2.2,dither=true))
f1=f3kdb(15,80,80,80,80,80, dither_algo=2, keep_tv_range=true, random_algo_ref=2)
f2=Fillborder(0,0,1,0)
f3=f3kdb(15,64,32,32,48,16, dither_algo=2, keep_tv_range=true, random_algo_ref=2)
f4=StackVertical(s.crop(0,0,0,6).FixRowBrightness(1,16,140,16,235).FillBorder(0,1,0,0).crop(0,0,0,2).Tweak(sat=2,dither=true),s.FixRowBrightness(1037,16,230,16,235).crop(0,2,0,1036),s.FixRowBrightness(1038,16,120,16,235).crop(0,1034,0,0).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=2.2,dither=true)).FixColumnBrightness(1915,16,235,16,233).FixColumnBrightness(1917,16,235,16,220)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 156 342 1
R 370 777 2
R 9510 9618 3
R 31460 31497 3
R 31561 31591 3
R 84287 84308 3
R 127291 128049 3
R 167157 167259 4
