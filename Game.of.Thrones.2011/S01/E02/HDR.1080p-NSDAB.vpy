import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf

src=core.ffms2.Source("")

src=fvf.Depth(src,16)

src=fvf.Debilinear(src,1920,1080,yuv444=True)

b16=fvf.Depth(src,16)

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(b16)
merged = core.std.MaskedMerge(db, b16, mask)
sfdb=fvf.ReplaceFramesSimple(b16, merged, mappings="[135 191]")

#tonemap=fvf.Depth(sfdb,32)
#tonemap=core.tonemap.Reinhard(tonemap,0.8,0.5,0.9)

final=fvf.Depth(sfdb,10)

final.set_output()