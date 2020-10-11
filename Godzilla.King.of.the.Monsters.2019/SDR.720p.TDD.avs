FFVideoSource("F:\Workshop\GodzillaKingofMonsters\2-video.mkv").Crop(0, 140, -0, -140).z_Spline36Resize(1280, 534, dither="error_diffusion")
db00 = f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=32, grainC=32, dynamic_grain=True, keep_tv_range=True)
dbdark = mt_merge(last, f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=32, grainC=32, dynamic_grain=True), mt_binarize(36, true), u=3, v=4)
GConditionalFilter(db00, last, "ST", "==", "1")
GConditionalFilter(dbdark, last, "ST", "==", "2")
ConditionalReader("filter.txt", "ST", show=False)

----------
filter.txt

type int
default 0

R 1735 1755 1
R 11561 11632 2
R 14689 14725 1
R 19939 19984 2
R 21456 21492 2
R 25884 26009 2
R 48529 48560 2
R 57052 57126 2
R 57548 57709 2
R 61405 61633 2
R 62309 62322 2
R 62343 62410 2
R 62842 62958 2
R 79218 79627 2
R 79878 80054 2
R 87857 87940 2
R 117136 117499 2
R 128221 128303 2
R 141943 142001 2
R 149588 149653 1
R 164930 165044 2
R 169165 169284 1