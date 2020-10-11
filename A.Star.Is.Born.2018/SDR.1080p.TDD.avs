FFVideoSource("D:\Workshop\AStarIsBorn\2-video.mkv").Crop(0, 140, -0, -140)
FixRBr(0, 4).FixRBr(799, 4).FixCBr(0, 4).FixCBr(1919, 4).bbmod(1, 1, 1, 1)
db01 = last.f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=64, grainC=64, dynamic_grain=True)
db02 = StackHorizontal(Crop(0, 0, 2, 0), Crop(2, 0, -2, 0).FillBorder(2, 0, 2, 0), Crop(1918, 0, 0, 0)).FillBorder(2, 0, 2, 0).f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=64, grainC=64, dynamic_grain=True)
last
ConditionalFilter(db01, last, "ST", "==", "1")
ConditionalFilter(db02, last, "ST", "==", "2")
ConditionalReader("filter.txt", "ST", show=False)

---------
Filter.txt

type int
default 0

R 64 243 1
R 287 466 2
R 52228 52354 1
R 98977 99097 1