DGDecodeNV_DGSource("F:\Workshop\ABeautifulMind\2-video.dgi").Crop(0, 20, -0, -20).FillBorders(0, 1, 0, 1)
dl01 = Fixer(3, 0, 3, 0, luma=false, chroma=true).FixCBr(3, 6).FixCBr(2, -5).FixCBr(1, 5).FixCBr(0, 40).FixCBr(1916, 6).FixCBr(1917, -5).FixCBr(1918, 10).FixCBr(1919, 40).bbmod(0, 0, 2, 2, blur=999, full_range=false).z_Spline36Resize(1280, 694, dither="error_diffusion")
z_Spline36Resize(1280, 694, dither="error_diffusion")
dl02 = Fixer(3, 0, 0, 0, luma=false).FillBorders(2, 0, 0, 0).FixCBr(1279, 40).bbmod(0, 0, 4, 2, blur=60)
ConditionalSelect("ST", last, dl01, dl02)
ConditionalReader("dl_filter.txt", "ST", show=False)

-------------
dl_filter.txt

type int
default 0

R 0 31833 1
R 31834 32555 2
R 32556 194948 1