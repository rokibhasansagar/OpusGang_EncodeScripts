FFVideoSource()
Crop(0, 138, -0, -138)
f1=FixColumnBrightness(2,16,178,16,255).FixColumnBrightness(1,16,100,16,255).FillMargins(1,0,0,0).bbmod(0,0,2,0)
f2=FixColumnBrightness(1919,16,210,16,255)
f3=FixColumnBrightness(1918,16,220,16,253).FillMargins(0,0,1,0).bbmod(0,0,0,2)
f4=Stackhorizontal(crop(0,0,1912,0),FixColumnBrightness(1912,16,230,16,255).FixColumnBrightness(1913,16,110,16,255).SmoothTweak(saturation=1.3).crop(1912,0,2,0),crop(1914,0,6,0))
f5=FillMargins(0,1,0,0)
f6=FixRowBrightness(3,16,250,16,255).FixRowBrightness(1,16,242,16,255).FixRowBrightness(0,16,255,16,250)
f8=FixColumnBrightness(2,16,210,16,255).FixColumnBrightness(1,16,100,16,255).FillMargins(1,0,0,0).bbmod(0,0,2,0)
f9=FillMargins(1,0,1,0).bbmod(0,0,2,2)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalFilter(f5, last, "SceneType", "=", "5")
ConditionalFilter(f6, last, "SceneType", "=", "6")
ConditionalFilter(f8, last, "SceneType", "=", "7")
ConditionalFilter(f9, last, "SceneType", "=", "8")
ConditionalReader("...fix.txt", "SceneType")
Spline36Resizemod(1280, 536)
f7=crop(0,0,-2,0).AddBorders(2,0,0,0)
ConditionalFilter(f7, last, "SceneType", "=", "4")
ConditionalReader("...fix.txt", "SceneType")
Trim(f3kdb(15,64,64,64,10,10),0,792)+Trim(793,0)

fix.txt

type int
default 0

R 0 225 1
R 226 333 7
R 334 527 8
R 32063 32076 2
R 32103 32137 2
R 32189 32217 2
R 32282 32337 2
R 32353 32370 2
R 32410 32427 2
R 32442 32456 2
R 76314 76374 3
R 76411 76545 3
R 76581 76601 3
R 77632 77719 4
R 78226 78431 4
R 99836 99866 5
R 99949 99960 5
R 100012 100029 5
R 110815 111066 3
R 119888 119917 6
R 119958 120041 6
R 134488 134529 6
R 135990 136043 6
