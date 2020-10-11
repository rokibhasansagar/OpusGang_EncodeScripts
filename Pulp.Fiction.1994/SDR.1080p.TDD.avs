FFVideoSource("F:\Workshop\PulpFiction\2-video.mkv").Crop(0, 132, -0, -132)
aa00 = maa2(ss=4.0, aa=64)
db00 = f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=32, grainC=32, dynamic_grain=True)
db01 = mt_merge(last, f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=32, grainC=32, dynamic_grain=True), mt_binarize(36, true), u=3, v=4)
GConditionalFilter(aa00, last, "ST", "==", "1")
GConditionalFilter(db00, last, "ST", "==", "2")
GConditionalFilter(db01, last, "ST", "==", "3")
ConditionalReader("filter.txt", "ST", show=False)

----------
filter.txt

type int
default 0

R 25 311 2
R 333 604 1
R 30352 30467 1
R 37593 37916 3
R 38039 38085 3
R 45052 45756 3
R 71674 71807 3
R 78160 78184 3
R 91212 91233 3
R 98884 98926 3
R 98958 99078 1