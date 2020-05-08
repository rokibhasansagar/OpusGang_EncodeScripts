import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import vsTAAmbk as taa
import sgvsfunc as sgf
from debandmask import *
from vsutil import *

#progress 5772

core.max_cache_size=15000

src = core.ffms2.Source("")

src = fvf.Depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

db = src.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(src)
merged = core.std.MaskedMerge(db, src, mask)
sfdb = fvf.ReplaceFramesSimple(src, merged, mappings="[135 191]")

db = src.f3kdb.Deband(range=10, y=50, cb=0, cr=0, grainy=64, grainc=15, output_depth=16, dynamic_grain=True)
mask = kgf.kirsch(get_y(src)).std.Maximum().std.Binarize(5000).std.Inflate() # This is basically a faster retinex mask with worse results in dark areas.  The debanded stuff here isn't in dark enough areas for there to be a big difference, though.
merged = core.std.MaskedMerge(db, src, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings="[7486 7489] [7580 7626] [8108 8119] [8554 8663] [8793 8932] [10730 10895] [10936 11075] [11367 11602] [12375 12430] [37942 37964]")

final = fvf.Depth(sfdb, 10)

final.set_output()