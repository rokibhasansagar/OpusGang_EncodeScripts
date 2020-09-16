FFVideoSource()
Crop(0, 138, -0, -138)
FixRBr(0,-5)
bbmod(1,0,0,0,thresh=8)
bbmod(0,1,0,0,thresh=20)
f1=FixCBr(1914,-5).FixCBr(1914,-2,30).FixCBr(1915,-100)
f2=FixCBr(1918,-3).FillBorder(0,0,1,0).Fixer(0,0,1,0).bbmod(0,0,0,2,thresh=16)
f3=f3kdb(15,64,48,48,32,16,keep_tv_range=true,dither_algo=2,random_algo_ref=2,random_algo_grain=2)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalReader("fix.txt", "SceneType")

fix.txt

type int
default 0

R 3600 3676 1
R 16164 17048 2
R 71608 71848 2
R 73939 74216 3
R 74241 74371 3
R 74443 74472 3
R 74910 75007 3
R 75093 75148 3
R 75333 75432 3
R 75513 75567 3
R 75656 75691 3
R 75762 75840 3
R 75888 75916 3
R 75979 76012 3
R 76298 76330 3
R 78280 78317 3
R 78764 78787 3
R 79855 79878 3
R 83146 83308 3
R 101857 101930 3
