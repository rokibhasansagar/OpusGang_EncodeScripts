FFVideoSource("F:\Workshop\Brightburn\2-video.mkv").Crop(0, 138, -0, -138)
Trim(0, 900) + Trim(901, 1188).bbmod(2, 0, 3, 3, blur=100) + Trim(1189, 0).bbmod(2, 0, 0, 0, blur=100)
db01 = f3kdb(range=15, Y=48, Cb=48, Cr=48, grainY=32, grainC=32, dynamic_grain=True, keep_tv_range=True)
db02 = f3kdb(range=15, Y=70, Cb=70, Cr=70, grainY=70, grainC=70, dynamic_grain=True, keep_tv_range=True)
GConditionalFilter(db01, last, "ST", "==", "1")
GConditionalFilter(db02, last, "ST", "==", "2")
ConditionalReader("filter.txt", "ST", show=False)

----------
filter.txt

type int
default 0

R 901 1188 1
R 38546 38601 1
R 47432 47502 1
R 59516 59575 1
R 106153 106173 1
R 106243 106294 1
R 110312 110932 1
R 112315 112659 1
R 112845 112875 2