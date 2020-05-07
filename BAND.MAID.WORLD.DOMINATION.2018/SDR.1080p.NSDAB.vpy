import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
from adptvgrnMod import *

src = core.ffms2.Source("")

src_16 = fvf.Depth(src, 16)

retinex = kgf.retinex_edgemask(src_16)

deband = core.f3kdb.Deband(src_16, range=30, y=56, cr=64, cb=64, grainy=32, grainc=32, dynamic_grain=True, output_depth=16, keep_tv_range=True)
dbmask = retinex.std.Maximum().std.Inflate().std.Inflate()
dbmerge = core.std.MaskedMerge(deband, src_16, dbmask)
dbmerge = adptvgrnMod(dbmerge, strength=0.1, size=1.5, sharp=50, static=False, grain_chroma=False, luma_scaling=10)

ringmask_in = core.std.Maximum(retinex).std.Maximum()
ringmask_de = core.std.Minimum(retinex)
ringmask = core.std.Expr([ringmask_in, ringmask_de], "x y -")

dering = haf.HQDeringmod(dbmerge, ringmask=ringmask, mrad=0, thr=20, msmooth=0, drrep=0)

out = fvf.Depth(dering, 10)

out.set_output()