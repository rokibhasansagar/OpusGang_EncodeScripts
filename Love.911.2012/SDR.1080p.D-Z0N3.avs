FFVideoSource()
Crop(0, 130, -0, -130)
ss=last
StackHorizontal(crop(0,0,6,0).FixColumnBrightness(1,16,180,16,235).FillBorder(1,0,0,0).crop(0,0,2,0).Tweak(sat=3.8,dither=true),crop(2,0,2,0).Tweak(sat=1.3,dither=true),crop(4,0,1908,0),crop(1912,0,2,0).Tweak(sat=1.1,dither=true),crop(1914,0,2,0),crop(1916,0,2,0).Tweak(sat=1.3,dither=true),\
crop(1914,0,0,0).FixColumnBrightness(4,16,181,16,235).FillBorder(0,0,1,0).crop(4,0,0,0).Tweak(sat=3.8,dither=true))
w=last
StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,816),crop(0,818,0,0).FixRowBrightness(1,16,170,16,235).Tweak(sat=1.2,dither=true))
s=last

f30=f3kdb(15,64,48,48,16,16,keep_tv_range=true,random_algo_ref=2,random_algo_grain=2)

w
f1=StackVertical(crop(0,0,0,6).FixRowBrightness(2,16,235,16,228).FixRowBrightness(1,16,126,16,235).FillBorder(0,1,0,0),crop(0,6,0,808),crop(0,814,0,0).FixRowBrightness(3,16,235,16,229).FixRowBrightness(4,16,126,16,235).FillBorder(0,0,0,1))
f2=StackVertical(crop(0,0,0,4).FixRowBrightness(2,16,205,16,235).FixRowBrightness(1,16,235,16,16),crop(0,4,0,810),crop(0,814,0,4).FixRowBrightness(2,16,235,16,230).FixRowBrightness(3,16,220,16,235),crop(0,814,0,0).FixRowBrightness(4,16,160,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=2,dither=true))
f3=StackVertical(crop(0,0,0,6).FixRowBrightness(4,16,230,16,235).FixRowBrightness(3,16,235,16,230).FixRowBrightness(2,16,197,16,235).FixRowBrightness(1,16,235,16,16),crop(0,6,0,812),crop(0,814,0,0).FixRowBrightness(4,16,195,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=2,dither=true))
f4=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,220,16,235).FixRowBrightness(0,16,112,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,816),crop(0,818,0,0).FixRowBrightness(0,16,235,16,235).FixRowBrightness(1,16,170,16,235).Tweak(sat=1.2,dither=true))
f5=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,816),crop(0,814,0,0).FixRowBrightness(4,16,210,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=1.5,dither=true))
f6=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,814).Fixer(0,0,0,1),crop(0,816,0,2),crop(0,814,0,0).FixRowBrightness(4,16,210,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=1.5,dither=true))
f7=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,814).Fixer(0,0,0,1),crop(0,816,0,2).FixRowBrightness(1,16,232,16,235),crop(0,814,0,0).FixRowBrightness(4,16,215,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=1.5,dither=true))
f8=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,814).Fixer(0,0,0,1),crop(0,816,0,2).FixRowBrightness(0,16,235,16,232).FixRowBrightness(1,16,230,16,235),crop(0,814,0,0).FixRowBrightness(4,16,220,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).\
Tweak(sat=1.5,dither=true))
f9=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,814).Fixer(0,0,0,1),crop(0,816,0,2).FixRowBrightness(0,16,235,16,228).FixRowBrightness(1,16,230,16,235),crop(0,814,0,0).FixRowBrightness(4,16,225,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).
\Tweak(sat=1.5,dither=true))
f10=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,814).Fixer(0,0,0,1),crop(0,816,0,2).FixRowBrightness(0,16,235,16,228).FixRowBrightness(1,16,230,16,235),crop(0,814,0,0).FixRowBrightness(4,16,233,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).\
Tweak(sat=1.5,dither=true))
f11=StackVertical(crop(0,0,0,4).FixRowBrightness(1,16,235,16,16).FixRowBrightness(2,16,233,16,235).FixRowBrightness(3,16,235,16,228),crop(0,4,0,814).FixRowBrightness(813,16,235,16,230),crop(0,814,0,0).FixRowBrightness(4,16,200,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=2,dither=true))
f13=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,804),crop(0,806,0,4).FixRowBrightness(2,16,235,16,230).Fixer(0,0,0,1),crop(0,810,0,0))
f14=StackVertical(crop(0,0,0,2),crop(0,2,0,2).FixRowBrightness(1,16,235,16,230).Fixer(0,1,0,0),crop(0,4,0,812),crop(0,814,0,0).FixRowBrightness(3,16,235,16,230).FixRowBrightness(4,16,200,16,235).FillBorder(0,0,0,1).crop(0,2,0,0))
f15=StackVertical(crop(0,0,0,2),crop(0,2,0,2).FixRowBrightness(1,16,235,16,230).Fixer(0,1,0,0),crop(0,4,0,814),crop(0,818,0,0).FixRowBrightness(1,16,170,16,235).Tweak(sat=1.2,dither=true))
f16=StackVertical(crop(0,0,0,6).Fixer(0,3,0,0,chroma=true).FixRowBrightness(0,16,235,16,16).crop(0,0,0,2),crop(0,2,0,2).FixRowBrightness(1,16,225,16,235).Fixer(0,1,0,0),crop(0,4,0,814),crop(0,818,0,0).FixRowBrightness(1,16,170,16,235).Tweak(sat=1.2,dither=true))
f17=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,808),crop(0,810,0,0).FixRowBrightness(1,16,230,16,235).FixRowBrightness(3,16,233,16,235).FixRowBrightness(4,16,235,16,16))
f18=StackVertical(crop(0,0,0,2),crop(0,2,0,2).FixRowBrightness(1,16,235,16,230).Fixer(0,1,0,0),crop(0,4,0,810),crop(0,814,0,0).FixRowBrightness(2,16,228,16,235).FixRowBrightness(3,16,235,16,210).Fixer(0,0,0,2,chroma=true).FillBorder(0,0,0,1))
f19=StackVertical(crop(0,0,0,2),crop(0,2,0,2).FixRowBrightness(1,16,235,16,230).Fixer(0,1,0,0),crop(0,4,0,808),crop(0,812,0,0).FillBorder(0,0,0,1).Fixer(0,0,0,6,8,chroma=true,cbottom=1))
f20=StackVertical(crop(0,0,0,6).FixRowBrightness(4,16,230,16,235,168).FillBorder(0,1,0,0).Fixer(0,2,0,0,4,chroma=true),crop(0,6,0,808),crop(0,814,0,0).FixRowBrightness(1,16,230,16,235).Fixer(0,0,0,3))
f21=StackVertical(crop(0,0,0,6).FixRowBrightness(5,16,230,16,235,168).FixRowBrightness(3,16,228,16,235).FillBorder(0,1,0,0).Fixer(0,3,0,0,chroma=true),crop(0,6,0,808),crop(0,814,0,0).FixRowBrightness(1,16,235,16,228).Fixer(0,0,0,3))
f22=StackVertical(crop(0,0,0,6).FixRowBrightness(2,16,235,16,233).Fixer(0,2,0,0,chroma=true),crop(0,6,0,810),Fixer(0,0,0,3).crop(0,816,0,0))
f23=StackVertical(crop(0,0,0,6).Fixer(0,3,0,0,chroma=true).FillBorder(0,1,0,0),crop(0,6,0,810),Fixer(0,0,0,3).crop(0,816,0,0))
f24=StackVertical(crop(0,0,0,6).Fixer(0,3,0,0,chroma=true).FillBorder(0,1,0,0),crop(0,6,0,808),crop(0,814,0,0).FixRowBrightness(1,16,235,16,230).Fixer(0,0,0,3))
f25=StackVertical(crop(0,0,0,2),crop(0,2,0,2).FixRowBrightness(1,16,235,16,230).FixRowBrightness(0,16,225,16,235),crop(0,4,0,812),crop(0,814,0,0).FixRowBrightness(3,16,235,16,230).Fixer(0,0,0,2,chroma=true).FillBorder(0,0,0,1).crop(0,2,0,0))
f26=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,17),crop(0,2,0,0).Fixer(0,2,0,0).crop(0,0,0,814),crop(0,814,0,0).Fixer(0,0,0,3,chroma=true).FillBorder(0,0,0,1).crop(0,2,0,0))
f27=StackVertical(crop(0,0,0,6).FixRowBrightness(5,16,235,16,17),crop(0,8,0,0).Fixer(0,2,0,0).crop(0,0,0,2),crop(0,8,0,2).FixRowBrightness(1,16,235,16,233).FixRowBrightness(0,16,233,16,235),crop(0,10,0,808),crop(0,818,0,0).FixRowBrightness(0,16,235,16,232).FixRowBrightness(1,16,175,16,235))
f28=Fixer(0,1,0,1)
f29=f2.f3kdb(15,64,48,48,16,16,keep_tv_range=true,random_algo_ref=2,random_algo_grain=2)

ss
StackHorizontal(crop(0,0,6,0).FixColumnBrightness(1,16,180,16,235).FillBorder(1,0,0,0).crop(0,0,2,0).Tweak(sat=3.8,dither=true),crop(2,0,2,0).Tweak(sat=1.3,dither=true),crop(4,0,1904,0).FixColumnBrightness(1903,16,233,16,235),crop(1908,0,0,0).FixColumnBrightness(0,16,232,16,235).FixColumnBrightness(2,16,233,16,235).\
FixColumnBrightness(3,16,225,16,235).FixColumnBrightness(4,16,235,16,225).FixColumnBrightness(5,16,233,16,235).FixColumnBrightness(6,16,105,16,235))
f12=StackVertical(crop(0,0,0,2).FixRowBrightness(1,16,235,16,232).FixRowBrightness(0,16,174,16,235).Tweak(sat=1.2,dither=true),crop(0,2,0,816),crop(0,818,0,0).FixRowBrightness(1,16,170,16,235).Tweak(sat=1.2,dither=true))
ConditionalFilter(f1, s, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalFilter(f5, last, "SceneType", "=", "5")
ConditionalFilter(f6, last, "SceneType", "=", "6")
ConditionalFilter(f7, last, "SceneType", "=", "7")
ConditionalFilter(f8, last, "SceneType", "=", "8")
ConditionalFilter(f9, last, "SceneType", "=", "9")
ConditionalFilter(f10, last, "SceneType", "=", "10")
ConditionalFilter(f11, last, "SceneType", "=", "11")
ConditionalFilter(f12, last, "SceneType", "=", "12")
ConditionalFilter(f13, last, "SceneType", "=", "13")
ConditionalFilter(f14, last, "SceneType", "=", "14")
ConditionalFilter(f15, last, "SceneType", "=", "15")
ConditionalFilter(f16, last, "SceneType", "=", "16")
ConditionalFilter(f17, last, "SceneType", "=", "17")
ConditionalFilter(f18, last, "SceneType", "=", "18")
ConditionalFilter(f19, last, "SceneType", "=", "19")
ConditionalFilter(f20, last, "SceneType", "=", "20")
ConditionalFilter(f21, last, "SceneType", "=", "21")
ConditionalFilter(f22, last, "SceneType", "=", "22")
ConditionalFilter(f23, last, "SceneType", "=", "23")
ConditionalFilter(f24, last, "SceneType", "=", "24")
ConditionalFilter(f25, last, "SceneType", "=", "25")
ConditionalFilter(f26, last, "SceneType", "=", "26")
ConditionalFilter(f27, last, "SceneType", "=", "27")
ConditionalFilter(f28, last, "SceneType", "=", "28")
ConditionalFilter(f29, last, "SceneType", "=", "29")
ConditionalFilter(f30, last, "SceneType", "=", "30")
ConditionalReader("fix.txt", "SceneType")

fix.txt

type int
default 0

R 0 339 1
R 340 416 2
R 417 620 29
R 620 624 30
R 19137 19331 3
R 19412 19511 3
R 19566 19636 3
R 20557 21026 3
R 24251 24292 3
R 24602 24656 4
R 24703 24748 4
R 24776 24826 4
R 24899 24963 4
R 25013 25047 4
R 25085 25118 4
R 25148 25211 4
34476 3
R 34477 34478 5
R 34479 34480 6
R 34481 34484 7
R 34485 34486 8
34487 9
R 34488 34489 10
34490 11
R 34635 34689 12
R 38618 38698 13
R 40115 40226 14
R 49237 49280 15
R 49764 49957 16
R 66501 66555 17
R 74083 74368 18
R 77439 77583 19
R 93868 94405 20
R 94406 94507 21
R 94627 94748 22
R 94749 94809 23
R 95474 95556 24
R 96756 96815 25
R 97434 97562 25
R 111883 112243 26
R 134255 134350 27
R 154089 154116 25
R 168449 168537 28
