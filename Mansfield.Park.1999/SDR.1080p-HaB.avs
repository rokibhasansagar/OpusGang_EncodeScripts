a0=FFVideoSource().Crop(0, 20, -0, -20).FixColumnBrightness(1913,16,230,16,235).FixColumnBrightness(1914,16,240,16,235).FixColumnBrightness(1915,16,233,16,235).FixColumnBrightness(1916,16,230,16,235)
a=a0.FixColumnBrightness(6,16,232,16,235).FixColumnBrightness(3,16,225,16,235).FixColumnBrightness(2,16,250,16,235).FixColumnBrightness(1,16,205,16,235).FixColumnBrightness(0,16,180,16,235)
a1=a0.FixColumnBrightness(6,16,232,16,235).FixColumnBrightness(3,16,230,16,235).FixColumnBrightness(2,16,250,16,235).FixColumnBrightness(1,16,205,16,235).FixColumnBrightness(0,16,180,16,235)
b=a.FixColumnBrightness(1917,16,255,16,235).FixColumnBrightness(1918,16,195,16,235).FixColumnBrightness(1919,16,150,16,235)
b1=a1.FixColumnBrightness(1917,16,250,16,235).FixColumnBrightness(1918,16,190,16,235).FixColumnBrightness(1919,16,150,16,235)
s=b.FixRowBrightness(1036,16,237,16,235).fillmargins(1,1,1,1).BalanceBorders(1,1,2,1)
s2=a.FixColumnBrightness(1917,16,260,16,235).FixColumnBrightness(1918,16,195,16,235).FixColumnBrightness(1919,16,150,16,235).FixRowBrightness(1036,16,237,16,235).fillmargins(1,1,1,1).BalanceBorders(1,1,2,1)
s3=b.FixRowBrightness(1035,16,265,16,235).FixRowBrightness(1036,16,225,16,235).fillmargins(1,1,1,1).BalanceBorders(1,1,2,1)
s4=b.FixRowBrightness(1036,16,237,16,235).FixRowBrightness(1038,16,225,16,235).fillmargins(1,1,1,1).BalanceBorders(1,1,2,1)
s5=b1.FixRowBrightness(1036,16,237,16,235).fillmargins(1,1,1,1).BalanceBorders(1,1,2,1)
StackVertical(StackHorizontal(s.smoothtweak(saturation=1.8).crop(0,0,2,0),s.smoothtweak(saturation=1.4).crop(2,0,2,0),s.crop(4,0,1912,0),s.smoothtweak(saturation=1.3).crop(1916,0,2,0),s.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,0,0,2),StackHorizontal(s.smoothtweak(saturation=1.8)
\.crop(0,0,2,0),s.smoothtweak(saturation=1.4).crop(2,0,2,0),s.crop(4,0,1912,0),s.smoothtweak(saturation=1.3).crop(1916,0,2,0),s.smoothtweak(saturation=1.9).crop(1918,0,0,0)).crop(0,2,0,1036),StackHorizontal(s.smoothtweak(saturation=1.8).crop(0,0,2,0),s.smoothtweak(saturation=1.4).crop(2,0,2,0),s.crop(4,0,1912,0),
\s.smoothtweak(saturation=1.3).crop(1916,0,2,0),s.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,1038,0,0))
t0=StackVertical(StackHorizontal(s3.smoothtweak(saturation=1.8).crop(0,0,2,0),s3.smoothtweak(saturation=1.4).crop(2,0,2,0),s3.crop(4,0,1912,0),s3.smoothtweak(saturation=1.3).crop(1916,0,2,0),s3.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,0,0,2),StackHorizontal(s3.smoothtweak(saturation=1.8)
\.crop(0,0,2,0),s3.smoothtweak(saturation=1.4).crop(2,0,2,0),s3.crop(4,0,1912,0),s3.smoothtweak(saturation=1.3).crop(1916,0,2,0),s3.smoothtweak(saturation=1.9).crop(1918,0,0,0)).crop(0,2,0,1036),StackHorizontal(s3.smoothtweak(saturation=1.8).crop(0,0,2,0),s3.smoothtweak(saturation=1.4).crop(2,0,2,0),s3.crop(4,0,1912,0),
\s3.smoothtweak(saturation=1.3).crop(1916,0,2,0),s3.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,1038,0,0))
t1=t0.crop(0,0,0,-2).AddBorders(0,2,0,0)
t2=StackVertical(StackHorizontal(s2.smoothtweak(saturation=1.8).crop(0,0,2,0),s2.smoothtweak(saturation=1.4).crop(2,0,2,0),s2.crop(4,0,1912,0),s2.smoothtweak(saturation=1.3).crop(1916,0,2,0),s2.smoothtweak(saturation=3.5).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,0,0,2),StackHorizontal(s2.smoothtweak(saturation=1.8)
\.crop(0,0,2,0),s2.smoothtweak(saturation=1.4).crop(2,0,2,0),s2.crop(4,0,1912,0),s2.smoothtweak(saturation=1.3).crop(1916,0,2,0),s2.smoothtweak(saturation=3.5).crop(1918,0,0,0)).crop(0,2,0,1036),StackHorizontal(s2.smoothtweak(saturation=1.8).crop(0,0,2,0),s2.smoothtweak(saturation=1.4).crop(2,0,2,0),s2.crop(4,0,1912,0),
\s2.smoothtweak(saturation=1.3).crop(1916,0,2,0),s2.smoothtweak(saturation=3.5).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,1038,0,0))
t3=StackVertical(StackHorizontal(s.smoothtweak(saturation=1.8).crop(0,0,2,0),s.smoothtweak(saturation=1.4).crop(2,0,2,0),s.crop(4,0,1912,0),s.smoothtweak(saturation=1.3).crop(1916,0,2,0),s.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,0,0,2),StackHorizontal(s.smoothtweak(saturation=1.8)
\.crop(0,0,2,0),s.smoothtweak(saturation=1.4).crop(2,0,2,0),s.crop(4,0,1912,0),s.smoothtweak(saturation=1.3).crop(1916,0,2,0),s.smoothtweak(saturation=1.9).crop(1918,0,0,0)).crop(0,2,0,908),StackHorizontal(s2.smoothtweak(saturation=1.8)
\.crop(0,0,2,0),s2.smoothtweak(saturation=1.4).crop(2,0,2,0),s2.crop(4,0,1912,0),s2.smoothtweak(saturation=1.3).crop(1916,0,2,0),s2.smoothtweak(saturation=3.5).crop(1918,0,0,0)).crop(0,910,0,128),StackHorizontal(s.smoothtweak(saturation=1.8).crop(0,0,2,0),s.smoothtweak(saturation=1.4).crop(2,0,2,0),s.crop(4,0,1912,0),
\s.smoothtweak(saturation=1.3).crop(1916,0,2,0),s.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,1038,0,0))
t4=StackVertical(StackHorizontal(s4.smoothtweak(saturation=1.8).crop(0,0,2,0),s4.smoothtweak(saturation=1.4).crop(2,0,2,0),s4.crop(4,0,1912,0),s4.smoothtweak(saturation=1.3).crop(1916,0,2,0),s4.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,0,0,2),StackHorizontal(s4.smoothtweak(saturation=1.8)
\.crop(0,0,2,0),s4.smoothtweak(saturation=1.4).crop(2,0,2,0),s4.crop(4,0,1912,0),s4.smoothtweak(saturation=1.3).crop(1916,0,2,0),s4.smoothtweak(saturation=1.9).crop(1918,0,0,0)).crop(0,2,0,1036),StackHorizontal(s4.smoothtweak(saturation=1.8).crop(0,0,2,0),s4.smoothtweak(saturation=1.4).crop(2,0,2,0),s4.crop(4,0,1912,0),
\s4.smoothtweak(saturation=1.3).crop(1916,0,2,0),s4.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,1038,0,0))
t5=StackVertical(StackHorizontal(s5.smoothtweak(saturation=1.5).crop(0,0,2,0),s5.smoothtweak(saturation=1.4).crop(2,0,2,0),s5.crop(4,0,1912,0),s5.smoothtweak(saturation=1.3).crop(1916,0,2,0),s5.smoothtweak(saturation=1.9).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,0,0,2),StackHorizontal(s5.smoothtweak(saturation=1.5)
\.crop(0,0,2,0),s5.smoothtweak(saturation=1.4).crop(2,0,2,0),s5.crop(4,0,1912,0),s5.smoothtweak(saturation=1.3).crop(1916,0,2,0),s5.smoothtweak(saturation=2.7).crop(1918,0,0,0)).crop(0,2,0,1036),StackHorizontal(s5.smoothtweak(saturation=1.5).crop(0,0,2,0),s5.smoothtweak(saturation=1.4).crop(2,0,2,0),s5.crop(4,0,1912,0),
\s5.smoothtweak(saturation=1.3).crop(1916,0,2,0),s5.smoothtweak(saturation=2.7).crop(1918,0,0,0)).smoothtweak(saturation=1.7).crop(0,1038,0,0))
ConditionalFilter(last, t1, "SceneType", "<", "1")
ConditionalFilter(last, t2, "SceneType", "<", "2")
ConditionalFilter(last, t3, "SceneType", "<", "3")
ConditionalFilter(last, t4, "SceneType", "<", "4")
ConditionalFilter(last, t5, "SceneType", "<", "5")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 475 769 1
R 32096 32258 2
R 32768 32863 2
R 41371 41608 3
R 103397 103974 4
R 104228 104352 4
R 104469 104563 4
R 111800 111928 5
