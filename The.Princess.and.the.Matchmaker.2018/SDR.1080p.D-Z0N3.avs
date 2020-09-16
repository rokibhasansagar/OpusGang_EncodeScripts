FFVideoSource()
Crop(0, 138, -0, -138)
f1=Fixer(1,0,2,0)
f2=Fixer(2,0,2,0)
f3=f1.f3kdb(15,64,48,48,16,16,keep_tv_range=true,random_algo_ref=2,random_algo_grain=2)
f4=f2.f3kdb(15,64,48,48,16,16,keep_tv_range=true,random_algo_ref=2,random_algo_grain=2)
f5=f1.f3kdb(15,64,48,48,48,48,dither_algo=2,keep_tv_range=true,random_algo_ref=2,random_algo_grain=2)
f6=f2.f3kdb(15,64,48,48,48,48,dither_algo=2,keep_tv_range=true,random_algo_ref=2,random_algo_grain=2)
ConditionalFilter(f2, f1, "SceneType", "=", "1")
ConditionalFilter(f3, last, "SceneType", "=", "2")
ConditionalFilter(f4, last, "SceneType", "=", "3")
ConditionalFilter(f5, last, "SceneType", "=", "4")
ConditionalFilter(f6, last, "SceneType", "=", "5")
ConditionalReader("fix.txt", "SceneType")

fix.txt

type int
default 0

R 261 360 1
R 2078 2274 2
R 6269 6305 2
R 6889 7062 2
R 11040 11144 2
R 11678 11805 2
R 12899 12981 2
R 14116 14235 2
R 14503 14609 2
R 14660 14833 2
R 27754 27803 2
R 27901 27983 2
R 28100 28180 2
R 28326 28489 2
R 28552 28625 2
R 39343 39488 2
R 42499 42584 2
R 46748 46829 2
R 46989 47035 2
R 61215 62232 1
R 80958 81180 1
R 81181 81269 3
R 82148 82196 2
R 100621 100807 4
R 100808 102420 5
R 102421 102979 1
R 102980 102995 5
R 102996 103060 1
R 124254 124304 4
R 127571 127653 2
R 128068 128179 2
R 128356 128406 2
R 128437 128488 2
R 134598 134626 2
R 138408 138477 2
R 141977 142129 1
R 147210 147253 1
R 147254 147301 3
R 147302 148317 1
R 148318 148373 2
R 148455 148531 2
R 150466 150521 2
R 150875 150928 1
R 151483 151790 2
