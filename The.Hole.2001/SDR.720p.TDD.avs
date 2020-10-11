FFVideoSource("F:\Workshop\TheHole\2-video.mkv").Crop(0, 132, -0, -132)
Trim(0, 1183) + Trim(1184, 0).bbmod(2, 2, 0, 0)
dl00 = Crop(0, 6, 0, -6).bbmod(2, 2, 0, 0).AddBordersMod(0, 6, 0, 6, tsat=0.7, bsat=0.7)
ConditionalSelect("DT_ST", last, dl00)
ConditionalReader("dl_filter.txt", "DT_ST", show=False)
z_Spline36Resize(1280, 544, dither="error_diffusion")
db0 = neo_f3kdb(range=15, Y=80, Cb=64, Cr=64, grainY=48, grainC=32, dynamic_grain=True, keep_tv_range=True)
mask0 = debandmask(60,16,4,4,1).mt_expand(40).ExtractY().mt_invert()
db00 = mt_merge(last, db0, mask0)
ConditionalSelect("ST", last, db00)
ConditionalReader("filter.txt", "ST", show=False)

----------
filter.txt

type int
default 0

R 8 293 1
R 47393 47465 1
R 125314 125438 1
R 133103 133114 1
R 133733 133829 1

-------------
dl_filter.txt

type int
default 0

R 8 293 1