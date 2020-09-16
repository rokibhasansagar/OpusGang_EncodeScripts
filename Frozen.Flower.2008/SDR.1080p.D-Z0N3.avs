a=FFVideoSource().Crop(2, 132, -2, -132) #1px black + 1px quite blurred cropped on the both sides
aa=a.FixColumnBrightness(3,16,255,16,248).FixColumnBrightnessProtect(1,16,240,16,255,155).FixColumnBrightness(0,16,255,16,240).FixColumnBrightnessProtect(0,16,205,16,255,170)
b=aa.FixColumnBrightness(1911,16,250,16,255).FixColumnBrightness(1912,16,255,16,252).FixColumnBrightness(1913,16,250,16,255).FixColumnBrightness(1914,16,250,16,255).FixColumnBrightness(1915,16,255,16,240)
bb=StackHorizontal(b.SmoothTweak(saturation=1.4).crop(0,0,2,0),b.crop(2,0,1910,0),b.SmoothTweak(saturation=1.1).crop(1912,0,2,0),b.bbmod(0,0,0,1,thresh=6).SmoothTweak(saturation=1.4).crop(1914,0,2,0))
c=bb.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175).
\FixRowBrightness(811,16,253,16,255).FixRowBrightness(812,16,255,16,253).FixRowBrightnessProtect(813,16,250,16,255,60).FixRowBrightness(814,16,252,16,255).FixRowBrightnessProtectb(815,16,255,16,245,200)
StackVertical(c.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),c.crop(0,2,0,0))
f1=bb.FixRowBrightness(1,16,252,16,255).FixRowBrightnessProtectb(813,16,255,16,250,120).FixRowBrightnessProtect(813,16,252,16,255,40).FixRowBrightness(814,16,253,16,255)
d=aa.FixRowBrightness(1,16,252,16,255).FixRowBrightnessProtectb(813,16,255,16,250,120).FixRowBrightnessProtect(813,16,252,16,255,40).FixRowBrightness(814,16,253,16,255).FixColumnBrightness(1910,16,255,16,250).FixColumnBrightness(1911,16,253,16,255).FixColumnBrightness(1912,16,255,16,253).
\FixColumnBrightness(1913,16,255,16,248).FixColumnBrightness(1914,16,250,16,255).FixColumnBrightnessProtectb(1914,16,240,16,255,150).bbmod(0,0,0,2)
f2=StackHorizontal(d.SmoothTweak(saturation=1.4).crop(0,0,2,0),d.crop(2,0,0,0))
f3=StackHorizontal(aa.FixRowBrightness(1,16,252,16,255).FixRowBrightnessProtectb(813,16,255,16,250,120).FixRowBrightnessProtect(813,16,252,16,255,40).FixRowBrightness(814,16,253,16,255).FixColumnBrightness(1910,16,255,16,252).FixColumnBrightnessProtect(1911,16,255,16,240,145).
\FixColumnBrightnessProtect(1911,16,255,16,255,110).SmoothTweak(saturation=1.4).crop(0,0,2,0),aa.FixRowBrightness(1,16,252,16,255).FixRowBrightnessProtectb(813,16,255,16,250,120).FixRowBrightnessProtect(813,16,252,16,255,40).FixRowBrightness(814,16,253,16,255).FixColumnBrightness(1910,16,255,16,252).
\FixColumnBrightnessProtect(1911,16,255,16,240,145).FixColumnBrightnessProtect(1911,16,255,16,255,110).crop(2,0,1912,0).bbmod(0,0,0,2),aa.FixRowBrightnessProtectb(813,16,255,16,250,120).FixRowBrightnessProtect(813,16,252,16,255,40).FixRowBrightness(814,16,253,16,255).crop(1914,0,0,0))
da=bb.FixRowBrightness(9,16,255,16,253).FixRowBrightnessProtect(8,16,250,16,255,80).FixRowBrightness(7,16,253,16,255).FixRowBrightness(5,16,250,16,255).FixRowBrightness(4,16,255,16,245).FixRowBrightnessProtect(4,16,255,16,230,33).FixRowBrightnessProtect(4,16,255,16,230,70).FixRowBrightnessProtect(4,16,255,16,245,120).
\FixRowBrightnessProtect(3,16,255,16,235,35).FixRowBrightnessProtect(3,16,255,16,230,60).FixRowBrightnessProtect(3,16,255,16,240,120).FixRowBrightness(2,16,255,16,0).FixRowBrightness(810,16,250,16,255).FixRowBrightness(811,16,255,16,232).FixRowBrightnessProtect(811,16,255,16,240,70).
\FixRowBrightnessProtect(811,16,255,16,210,100).FixRowBrightnessProtect(811,16,240,16,255,120).FixRowBrightness(812,16,255,16,230).FixRowBrightnessProtect(812,16,255,16,200,95)
da1=StackVertical(da.crop(0,0,0,4).bbmod(0,1,0,0,thresh=3),da.crop(0,4,0,808).bbmod(0,1,0,0,thresh=8,blur=100),da.crop(0,812,0,0))
f4=StackVertical(da1.crop(0,0,0,814).bbmod(0,2,0,0),da1.crop(0,814,0,0)).FixRowBrightness(813,16,255,16,0)
dd=aa.FixColumnBrightness(1898,16,255,16,252).FixColumnBrightness(1901,16,255,16,248).FixColumnBrightness(1902,16,252,16,255).FixColumnBrightness(1904,16,255,16,230).FixColumnBrightnessProtect(1904,16,255,16,225,50).FixColumnBrightnessProtect(1904,16,255,16,240,110).FixColumnBrightnessProtect(1904,16,255,16,230,150).
\FixColumnBrightness(1905,16,245,16,255).FixColumnBrightnessProtect(1905,16,255,16,210,60).FixColumnBrightnessProtect(1905,16,255,16,220,150).FixColumnBrightness(1906,16,140,16,255).FixColumnBrightnessProtectb(1906,16,210,16,255,130)
dd1=StackHorizontal(dd.SmoothTweak(saturation=1.4).crop(0,0,2,0),dd.crop(2,0,1902,0).BalanceBorders(0,0,0,2),dd.crop(1904,0,0,0)).crop(0,0,-4,0).AddBorders(4,0,0,0)
dd2=dd1.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175).
\FixRowBrightness(811,16,253,16,255).FixRowBrightness(812,16,255,16,253).FixRowBrightnessProtect(813,16,250,16,255,60).FixRowBrightness(814,16,252,16,255).FixRowBrightnessProtectb(815,16,255,16,245,200)
f5=StackVertical(dd2.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),dd2.crop(0,2,0,0))
ddd=aa.FixColumnBrightness(1906,16,255,16,250).FixColumnBrightness(1908,16,250,16,255).FixColumnBrightness(1909,16,255,16,250).FixColumnBrightness(1910,16,250,16,255).FixColumnBrightness(1911,16,232,16,255).FixColumnBrightness(1912,16,255,16,247).
\FixColumnBrightness(1913,16,255,16,247).FixColumnBrightness(1914,16,155,16,255)
ddd1=StackHorizontal(ddd.SmoothTweak(saturation=1.4).crop(0,0,2,0),ddd.crop(2,0,1912,0).BalanceBorders(0,0,0,2,thresh=9,blur=50),ddd.crop(1914,0,0,0))
ddd2=ddd1.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175).
\FixRowBrightness(811,16,253,16,255).FixRowBrightness(812,16,255,16,253).FixRowBrightnessProtect(813,16,250,16,255,60).FixRowBrightness(814,16,252,16,255).FixRowBrightnessProtectb(815,16,255,16,245,200)
f6=StackVertical(ddd2.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),ddd2.crop(0,2,0,0))
e=a.FixColumnBrightness(6,16,255,16,252).FixColumnBrightness(5,16,255,16,252).FixColumnBrightness(2,16,240,16,255).FixColumnBrightness(1,16,220,16,255).FixColumnBrightness(0,16,220,16,255).BalanceBorders(0,0,2,0,thresh=16)
e1=e.FixColumnBrightness(1911,16,250,16,255).FixColumnBrightness(1912,16,255,16,252).FixColumnBrightness(1913,16,250,16,255).FixColumnBrightness(1914,16,250,16,255).FixColumnBrightness(1915,16,255,16,240)
e2=StackHorizontal(e1.crop(0,0,1912,0),e1.SmoothTweak(saturation=1.1).crop(1912,0,2,0),e1.bbmod(0,0,0,1,thresh=6).SmoothTweak(saturation=1.4).crop(1914,0,2,0))
e3=e2.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175).
\FixRowBrightness(811,16,253,16,255).FixRowBrightness(812,16,255,16,253).FixRowBrightnessProtect(813,16,250,16,255,60).FixRowBrightness(814,16,252,16,255).FixRowBrightnessProtectb(815,16,255,16,245,200)
f7=StackVertical(e3.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),e3.crop(0,2,0,0))
ee=bb.FixRowBrightness(802,16,250,16,255).FixRowBrightnessProtectb(805,16,248,16,255,190).FixRowBrightness(806,16,255,16,230).FixRowBrightnessProtect(806,16,255,16,230,45).FixRowBrightnessProtect(806,16,255,16,230,90).FixRowBrightness(807,16,255,16,250).FixRowBrightnessProtect(807,16,255,16,235,60).
\FixRowBrightnessProtect(807,16,255,16,205,110).FixRowBrightness(808,16,255,16,0)
ee1=ee.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175)
f8=StackVertical(ee1.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),ee1.crop(0,2,0,806).BalanceBorders(0,2,0,0,blur=50),ee1.crop(0,808,0,0)).crop(0,0,0,-4).AddBorders(0,4,0,0)
g=aa.FixColumnBrightness(1910,16,253,16,255).FixColumnBrightness(1911,16,252,16,255).FixColumnBrightness(1912,16,255,16,252).FixColumnBrightness(1913,16,245,16,255).FixColumnBrightness(1914,16,245,16,255).FixColumnBrightness(1915,16,255,16,238).BalanceBorders(0,0,0,2,thresh=9)
g1=StackHorizontal(g.SmoothTweak(saturation=1.4).crop(0,0,2,0),g.crop(2,0,0,0))
g2=g1.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175).
\FixRowBrightness(811,16,253,16,255).FixRowBrightness(812,16,255,16,253).FixRowBrightnessProtect(813,16,250,16,255,60).FixRowBrightness(814,16,252,16,255).FixRowBrightnessProtectb(815,16,255,16,245,200)
f9=StackVertical(g2.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),g2.crop(0,2,0,0))
gg=bb.FixRowBrightness(4,16,253,16,255).FixRowBrightness(3,16,255,16,253).FixRowBrightnessProtect(3,16,255,16,250,180).FixRowBrightnessProtect(2,16,250,16,255,75).FixRowBrightnessProtectb(1,16,250,16,255,210).FixRowBrightness(0,16,255,16,245).FixRowBrightnessProtect(0,16,245,16,255,90).FixRowBrightnessProtect(0,16,230,16,255,175).
\FixRowBrightness(811,16,253,16,255).FixRowBrightness(812,16,238,16,255).FixRowBrightness(813,16,248,16,255).FixRowBrightness(814,16,220,16,255)
f10=StackVertical(gg.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),gg.crop(0,2,0,812).bbmod(0,2,0,0,thresh=14),gg.crop(0,814,0,0)).bbmod(0,2,0,0,thresh=40).FixRowBrightness(815,16,255,16,0)
f11=StackVertical(gg.bbmod(1,0,0,0,thresh=5).SmoothTweak(saturation=1.1).crop(0,0,0,2),gg.crop(0,2,0,812).bbmod(0,2,0,0,thresh=14),gg.FixRowBrightness(815,16,220,16,255).crop(0,814,0,0)).bbmod(0,2,0,0,thresh=40)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalFilter(last, f7, "SceneType", "<", "7")
ConditionalFilter(last, f8, "SceneType", "<", "8")
ConditionalFilter(last, f9, "SceneType", "<", "9")
ConditionalFilter(last, f10, "SceneType", "<", "10")
ConditionalFilter(last, f11, "SceneType", "<", "11")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 0 963 1
R 964 1149 2
R 2338 2482 1
R 2483 2520 2
R 2521 2578 1
R 2579 2678 2
R 2679 2739 1
R 2740 2812 3
R 2813 2838 1
R 2839 2853 3
R 2854 2879 1
R 2880 3045 3
R 3046 3108 1
R 17394 17799 4
R 19552 19586 5
R 47259 47346 6
R 107101 107160 7
R 133704 133797 8
R 172624 172667 9
R 183183 183186 10
183187 11
R 183188 183201 10
R 183202 183205 11
R 183206 183210 10
183211 11
R 183212 183215 10
183216 11
R 183217 183225 10
R 183226 183227 11
R 183228 183249 10
183250 11
R 183251 183254 10
183255 11
183256 10
R 183531 183600 11
R 184101 184103 10
184104 11
R 184105 184182 10
R 185438 185449 10
185450 11
R 185451 185473 10
185474 11
R 185475 185498 10
