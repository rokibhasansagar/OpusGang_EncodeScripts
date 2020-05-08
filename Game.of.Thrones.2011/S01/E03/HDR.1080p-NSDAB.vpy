import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import vsTAAmbk as taa
import sgvsfunc as sgf

core.max_cache_size=15000

src = core.ffms2.Source("")

src = fvf.Depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

b16 = src

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(b16)
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(b16, merged, mappings="[135 191]")

db = b16.f3kdb.Deband(range=15, y=25, cb=0, cr=0, grainy=32, grainc=15, output_depth=16, dynamic_grain=True)
mask = kgf.retinex_edgemask(b16).std.Expr("x 2000 > x 10 * x ?")
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings="[63925 64174] [64272 64377] [62325 62470] [62630 62733] [62791 62867] [62961 63183] [63382 63994] [64784 64987]")

final = fvf.Depth(sfdb, 10)

final.set_output()