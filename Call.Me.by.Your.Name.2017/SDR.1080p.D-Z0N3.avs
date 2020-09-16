FFVideoSource().Crop(0, 20, -0, -20)
s=last
FillBorder(0,1,0,1)
StackVertical(crop(0,0,0,16).SmoothTweak(saturation=1.6,usemt=8).crop(0,0,0,2),crop(0,2,0,1036),crop(0,1024,0,0).SmoothTweak(saturation=2,usemt=8).crop(0,14,0,0))
f1=BalanceB(2,2,0,0)
f2a=s.FillBorder(1,1,0,1)
f2=StackVertical(f2a.crop(0,0,0,16).SmoothTweak(saturation=1.6,usemt=8).crop(0,0,0,2),f2a.crop(0,2,0,1036),f2a.crop(0,1024,0,0).SmoothTweak(saturation=2,usemt=8).crop(0,14,0,0))
f3=f3kdb(10,65,0,0,65,0,keep_tv_range=true,random_algo_ref=2)
f4=f3kdb(10,64,0,0,65,0,keep_tv_range=true,random_algo_ref=2)
f5=f3kdb(15,65,0,0,10,0,keep_tv_range=true,random_algo_ref=2)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalFilter(f5, last, "SceneType", "=", "5")
ConditionalReader("... fix.txt", "SceneType")

fix.txt

type int
default 0

R 49 236 1
R 334 468 2
R 7802 7932 3
R 8172 8251 3
R 9005 10778 4
R 39014 39060 3
R 41073 41105 3
R 42194 43115 5
R 55695 55795 3
R 59918 60205 3
R 100650 101729 4
R 107206 107577 3
R 108213 108323 4
R 112199 113172 4
R 142172 144448 3
R 153866 155051 3
R 159242 159307 3
R 183662 188645 4
R 189675 189862 1
