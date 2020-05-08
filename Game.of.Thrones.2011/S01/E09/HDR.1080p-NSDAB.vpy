import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
from vsutil import plane
from debandmask import *

src = core.ffms2.Source("")

src = fvf.Depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

db = src.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(src)
merged = core.std.MaskedMerge(db, src, mask)
sfdb = fvf.ReplaceFramesSimple(src, merged, mappings="[135 191]")

db = src.f3kdb.Deband(range=15, y=25, cb=15, cr=15, grainy=48, grainc=15, output_depth=16, dynamic_grain=True)
mask = debandmask(src).std.Inflate()
merged = core.std.MaskedMerge(db, src, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings="[3039 3860] [4076 5760] [6125 6538] [6660 6818] [6932 7025] [7115 7314] [7835 7894] [17199 17444] [19740 19807] [21640 21879] [39854 39895] [45532 45541] [46040 46132] [60979 61016] [62181 62262] [63414 63548] [70817 70862] [71586 71615] [79386 79391]")

final = fvf.Depth(sfdb, 10)

final.set_output()