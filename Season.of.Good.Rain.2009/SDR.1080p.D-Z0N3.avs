FFVideoSource()
Crop(0, 60, -0, -60)
s=last
StackHorizontal(crop(0,0,6,0).FixColumnBrightness(1,16,177,16,235).FillBorder(1,0,0,0).crop(0,0,2,0),crop(2,0,2,0).FixColumnBrightness(1,16,230,16,235,90).FixColumnBrightnessb(0,16,235,16,233,190).FixColumnBrightness(0,16,233,16,235,90).Tweak(sat=1.3,dither=true),\
crop(4,0,1906,0),crop(1910,0,6,0).FixColumnBrightness(0,16,233,16,235).FixColumnBrightness(3,16,227,16,235).FixColumnBrightness(4,16,235,16,225).FixColumnBrightness(5,16,230,16,235),crop(1900,0,18,0).FixColumnBrightness(16,16,225,16,235).FixColumnBrightness(17,16,235,16,205).FixColumnBrightnessb(17,16,221,16,235,208).\
crop(16,0,0,0).Tweak(sat=1.5,dither=true),crop(1914,0,0,0).FixColumnBrightness(4,16,205,16,235).FillBorder(0,0,1,0).crop(4,0,0,0))
s1=last
crop(1916,0,0,0)
a1=crop(2,0,0,0).mt_merge(last,mt_binarize(16,y=1,u=3,v=3),y=1,u=3,v=3)
s1
StackHorizontal(crop(0,0,4,0).BalanceB(0,0,2,0,thresh=4).crop(0,0,2,0).Tweak(sat=1.2,dither=true),crop(2,0,1916,0),a1)
s2=last

f0=f3kdb(15,64,32,32,48,16, dither_algo=2, keep_tv_range=true, random_algo_ref=2)

af1=StackHorizontal(crop(0,0,1914,0),crop(1914,0,0,0).FixColumnBrightness(0,16,233,16,235).crop(0,0,2,0).Tweak(sat=1.3,dither=true),crop(1916,0,0,0).FixColumnBrightness(1,16,200,16,235,180).BalanceB(0,0,0,3,thresh=6).Tweak(sat=1.3,dither=true))
f1=mt_merge(last, af1, mt_binarize(205), luma=true)

s
StackHorizontal(crop(0,0,6,0).FixColumnBrightness(1,16,177,16,235).FillBorder(1,0,0,0).crop(0,0,2,0),crop(2,0,2,0).FixColumnBrightness(1,16,230,16,235,90).FixColumnBrightnessb(0,16,235,16,233,190).FixColumnBrightness(0,16,233,16,235,90).Tweak(sat=1.3,dither=true),crop(4,0,0,0))
StackHorizontal(crop(0,0,4,0).BalanceB(0,0,2,0,thresh=4).crop(0,0,2,0).Tweak(sat=1.2,dither=true),crop(2,0,0,0))
l=last
StackHorizontal(crop(0,0,1910,0),crop(1910,0,6,0).FixColumnBrightness(0,16,233,16,235).FixColumnBrightness(3,16,227,16,235).FixColumnBrightness(4,16,235,16,230).FixColumnBrightness(5,16,230,16,235),crop(1900,0,18,0).FixColumnBrightness(16,16,225,16,235).FixColumnBrightness(17,16,235,16,205).\
FixColumnBrightness(17,16,221,16,235,130).crop(16,0,0,0).Tweak(sat=1.5,dither=true),crop(1914,0,0,0).FixColumnBrightness(4,16,205,16,235).FillBorder(0,0,1,0).crop(4,0,0,0))
sf2=last
crop(1916,0,0,0)
a1f2=crop(2,0,0,0).mt_merge(last,mt_binarize(16,y=1,u=3,v=3),y=1,u=3,v=3)
sf2
f2=StackHorizontal(crop(0,0,1918,0),a1f2)

l
StackHorizontal(crop(0,0,1910,0),crop(1910,0,6,0).FixColumnBrightness(0,16,233,16,235).FixColumnBrightness(3,16,227,16,235).FixColumnBrightness(4,16,235,16,230).FixColumnBrightness(5,16,230,16,235),crop(1900,0,18,0).FixColumnBrightness(16,16,225,16,235).FixColumnBrightness(17,16,235,16,205).\
FixColumnBrightness(17,16,221,16,235,170).crop(16,0,0,0).Tweak(sat=1.5,dither=true),crop(1914,0,0,0).FixColumnBrightness(4,16,205,16,235).FillBorder(0,0,1,0).crop(4,0,0,0))
sf3=last
crop(1916,0,0,0)
a1f3=crop(2,0,0,0).mt_merge(last,mt_binarize(16,y=1,u=3,v=3),y=1,u=3,v=3)
sf3
f3=StackHorizontal(crop(0,0,1918,0),a1f3)

s
StackHorizontal(crop(0,0,6,0).FixColumnBrightness(1,16,172,16,235).FillBorder(1,0,0,0).crop(0,0,2,0),crop(2,0,2,0).FixColumnBrightness(1,16,230,16,235,90).FixColumnBrightnessb(0,16,235,16,233,190).FixColumnBrightness(0,16,233,16,235,90).Tweak(sat=1.3,dither=true),crop(4,0,0,0))
StackHorizontal(crop(0,0,4,0).BalanceB(0,0,2,0,thresh=4).crop(0,0,2,0).Tweak(sat=1.2,dither=true),crop(2,0,0,0))
StackHorizontal(crop(0,0,1910,0),crop(1910,0,6,0).FixColumnBrightness(0,16,233,16,235).FixColumnBrightness(3,16,227,16,235).FixColumnBrightness(4,16,235,16,225).FixColumnBrightness(5,16,230,16,235),crop(1900,0,18,0).FixColumnBrightness(16,16,225,16,235).FixColumnBrightness(17,16,235,16,205).FixColumnBrightnessb(17,16,221,16,235,208).\
crop(16,0,0,0).Tweak(sat=1.5,dither=true),crop(1914,0,0,0).FixColumnBrightness(4,16,205,16,235).FillBorder(0,0,1,0).crop(4,0,0,0))
sf4=last
crop(1916,0,0,0)
a1f4=crop(2,0,0,0).mt_merge(last,mt_binarize(16,y=1,u=3,v=3),y=1,u=3,v=3)
sf4
f4=StackHorizontal(crop(0,0,1918,0),a1f4)

ConditionalFilter(f0, s2, "SceneType", "=", "1")
ConditionalFilter(f1, last, "SceneType", "=", "2")
ConditionalFilter(f2, last, "SceneType", "=", "3")
ConditionalFilter(f3, last, "SceneType", "=", "4")
ConditionalFilter(f4, last, "SceneType", "=", "5")
ConditionalReader("... fix.txt", "SceneType")

fix.txt

type int
default 0

R 46 796 1
R 16983 17065 1
R 17298 17528 1
R 31770 32013 1
R 35374 35493 1
R 63669 63713 1
R 64211 64254 1
R 71763 71819 1
R 80348 80374 2
R 96606 99373 3
R 110157 110197 4
R 119924 120157 1
R 120225 120382 1
R 120493 120642 1
R 120774 120939 1
R 121108 121385 1
R 128188 128195 1
R 133552 133660 5
