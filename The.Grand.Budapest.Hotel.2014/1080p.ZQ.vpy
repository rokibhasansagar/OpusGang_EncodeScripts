import vapoursynth as vs
core = vs.core

import kagefunc as kgf
import awsmfunc as awf
from vsutil import depth

core.max_cache_size = 4048
core.num_threads = 8

cc = core.ffms2.Source(r'E:\Source\budapest.cc.mkv').std.Crop(top=22, bottom=22).std.Trim(first=28)
us = core.ffms2.Source(r'E:\Source\The.Grand.Budapest.Hotel.2014.1080p.BluRay.Remux.AVC.DTS-HD.MA.5.1-TEPES.mkv').std.Crop(top=22, bottom=22)

# Replacing the hard subbed Criterion frames with the clean US frames
src = awf.rfs(cc, us, mappings="[35577 35682] [42380 42444] [45196 45242] [45298 45342] [45366 45433] [45439 45506]")
src = depth(src, 16)

# Variable crop on widerscreen scenes
wide = awf.fb(src, right=1, top=1, bottom=1, mode="repeat")
wide = awf.zr(wide, width=1920, left=2, right=1, top=1, bottom=1)
src = awf.rfs(src, wide, mappings="[4998 13414] [65281 66229] [130811 134306]")

mask = core.std.ShufflePlanes(src, 0, vs.GRAY).std.Binarize(43500)
cf = core.fb.FillBorders(src, top=1, mode="repeat").std.MaskedMerge(src, mask)
fill = awf.bbmod(cf, top=2, thresh=80, blur=45)

# Debanding
deband = core.neo_f3kdb.Deband(fill, range=25, y=48, cr=0, cb=0, grainy=55, grainc=55, output_depth=16, opt=3, dynamic_grain=True, sample_mode=4)
mask = kgf.retinex_edgemask(fill).std.Inflate()
merged = core.std.MaskedMerge(deband, fill, mask)
debanded = awf.rfs(fill, merged, filename=r'E:\Source\banded.txt')
final = depth(debanded, 8)
final.set_output()