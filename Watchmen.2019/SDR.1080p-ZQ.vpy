import vapoursynth as vs
import fvsfunc as fvf
import kagefunc as kgf
core = vs.core

src = core.ffms2.Source("/../Watchmen.S01E01.Its.Summer.and....of.Ice.1080p.BluRay.Remux.AVC.DTS-HD.MA.5.1-PmP.mkv")

b16 = fvf.Depth(src, 16)
db = core.f3kdb.Deband(b16, range=15, y=55, cr=0, cb=0, grainy=65, grainc=60, output_depth=16, dynamic_grain=True)
mask = kgf.retinex_edgemask(b16).std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
final = fvf.Depth(merged, 8)

final.set_output()
