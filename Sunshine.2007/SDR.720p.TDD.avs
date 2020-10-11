FFVideoSource("F:\Workshop\SunshineUSA\2-video.mkv").Crop(0, 136, -0, -136)
Fixer(bottom=2, luma=false, chroma=true).Fillborders(bottom=1)
z_Spline36Resize(1280, 538, dither="error_diffusion")
db_mask = debandmask(60,16,4,4,1).mt_expand(40).ExtractY().mt_invert()
db = neo_f3kdb(range=15, Y=64, Cb=64, Cr=64, grainY=48, grainC=0, dynamic_grain=true, keep_tv_range=true)
db00 = mt_merge(last, db, db_mask)
aa00 = maa2(ss=4.0, aa=48)
ConditionalSelect("ST", last, db00, aa00)
ConditionalReader("filter.txt", "ST", show=False)

----------
filter.txt

type int
default 0

R 24 3218 1
R 10680 10820 1
R 12361 12692 1
R 13222 13300 1
R 13384 13619 1
R 13882 14120 1
R 42709 42892 1
R 49485 49699 1
R 51505 51598 1
R 51928 51974 1
R 53192 53318 1
R 53553 53658 1
R 54145 54231 1
R 57227 57434 1
R 68183 68473 1
R 86218 86362 1
R 89785 89805 1
R 91547 91775 1
R 113078 113458 1
R 116306 116469 1
R 132578 132633 1
R 136196 136456 1
R 137670 137737 1
R 137835 137850 1
R 144519 144671 1
R 145169 154522 2