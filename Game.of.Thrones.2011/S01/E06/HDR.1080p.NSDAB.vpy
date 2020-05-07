import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
from debandmask import *

src=core.ffms2.Source("")

b16 = fvf.Depth(src, 16)

b16 = fvf.Debilinear(b16, 1920, 1080, yuv444=True)

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(b16)
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(b16, merged, mappings="[135 191]")

db = b16.f3kdb.Deband(range=15, y=25, cb=0, cr=0, grainy=32, grainc=15, output_depth=16, dynamic_grain=True)
mask = debandmask(b16).std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings="[2779 2824] [5956 6389] [6586 6881] [7307 7508] [7552 7905] [11338 11407] [55914 56015] [56153 56250] [63793 63942] [64102 64180] [64276 64393] [65438 65857] [65869 66108] [66146 66200] [66717 66792]")

final = fvf.Depth(sfdb, 10)

final.set_output()