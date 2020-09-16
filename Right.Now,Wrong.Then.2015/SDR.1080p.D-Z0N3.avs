FFVideoSource()
Crop(0, 20, -0, -20)
StackVertical(crop(0,0,0,6).FixRowBrightness(1,16,233,16,235).FillBorder(0,1,0,0).crop(0,0,0,2).Tweak(sat=2,dither=true),crop(0,2,0,1036),crop(0,1034,0,0).FixRowBrightness(4,16,233,16,235).FillBorder(0,0,0,1).crop(0,4,0,0).Tweak(sat=2,dither=true))
f1=f3kdb(15,64,48,48,16,16,dither_algo=2,keep_tv_range=true,random_algo_ref=2)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalReader("fix.txt", "SceneType")

type int
default 0

R 33427 33797 1
R 33859 35818 1
R 69062 72944 1
R 74968 75199 1
R 79809 80122 1
R 81207 82936 1
R 100460 100710 1
R 111359 111670 1
