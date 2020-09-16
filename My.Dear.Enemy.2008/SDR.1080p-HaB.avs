FFVideoSource()
Crop(0, 130, -0, -130)
StackVertical(FixRowBrightness(2,16,238,16,235).FixRowBrightness(1,16,148,16,235).smoothtweak(saturation=1.2).fillmargins(0,1,0,0).BalanceBorders(2,0,0,0,thresh=5,blur=10).crop(0,0,0,2),FixRowBrightness(2,16,238,16,235).FixRowBrightness(817,16,240,16,235).crop(0,2,0,816),FixRowBrightness(817,16,240,16,235).FixRowBrightness(818,16,150,16,235).
\smoothtweak(saturation=2.5).fillmargins(0,0,0,1).BalanceBorders(0,1,0,0).crop(0,818,0,2))
f1=FixColumnBrightness(1918,16,230,16,235).BalanceBorders(0,0,2,2,blur=1)
f2=FixColumnBrightness(1908,16,265,16,235).FixColumnBrightness(1909,16,290,16,235).FixColumnBrightness(1910,16,160,16,235).crop(0,0,-4,0).AddBorders(4,0,0,0)
f3=StackHorizontal(FixColumnBrightness(5,16,248,16,235).FixColumnBrightness(4,16,370,16,235).FixColumnBrightness(3,16,140,16,235).FixColumnBrightness(1914,16,240,16,235).FixColumnBrightness(1915,16,250,16,235).crop(0,0,-4,0).BalanceBorders(0,0,0,2,blur=20),FixColumnBrightness(1916,16,140,16,235).crop(1916,0,4,0))
f4=FixColumnBrightness(1912,16,405,16,235).FixColumnBrightness(1913,16,280,16,235).BalanceBorders(0,0,2,0,blur=1).crop(0,0,-2,0).AddBorders(2,0,0,0)
f5=StackHorizontal(FixColumnBrightness(9,16,265,16,235).FixColumnBrightness(8,16,330,16,235).FixColumnBrightness(7,16,200,16,235).crop(0,0,8,0),FixColumnBrightness(9,16,265,16,235).FixColumnBrightness(8,16,330,16,235).FixColumnBrightness(7,16,200,16,235).FixColumnBrightness(1918,16,230,16,235).BalanceBorders(0,0,0,2,blur=1)
\.crop(8,0,1912,0).BalanceBorders(0,0,2,0,blur=50)).crop(4,0,0,0).AddBorders(0,0,4,0)
f6=f1.f3kdb(grainY=0,grainC=0,dither_algo=3,y=65)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 0 53517 1
R 53518 53703 6
R 53704 67921 1
R 67922 68244 2
R 68245 93913 1
R 93914 95412 3
R 95413 132686 1
R 132687 132774 4
R 132775 151988 1
R 151989 152419 6
R 152420 171293 1
R 171294 172506 5
R 172627 174489 1
