import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf

src = core.ffms2.Source("")

src = fvf.Depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

b16 = fvf.Depth(src, 16)

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(b16)
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(b16, merged, mappings="[135 191]")

db = b16.f3kdb.Deband(range=15, y=25, cb=0, cr=0, grainy=32, grainc=15, output_depth=16, dynamic_grain=True)
mask = kgf.retinex_edgemask(b16).std.Binarize(10000)
mask2 = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Expr("x 4500 > x 5000 < and x 100 * x 100 / ?")
merged = core.std.MaskedMerge(b16, db, mask2)
merged = core.std.MaskedMerge(merged, b16, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings="[15590 15646] [27023 27173] [27276 27515] [27992 28075] [49255 49302]")

final = fvf.Depth(sfdb, 10)


final.set_output()