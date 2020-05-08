import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
from adptvgrnMod import *

src = core.ffms2.Source("Game.of.Thrones.S01E01.Winter.Is.Coming.2160p.UHD.BluRay.REMUX.HDR.HEVC.Atmos-EPSiLON.mkv")

src = fvf.Depth(src,16)

dss = fvf.Debilinear(src, 1920, 1080, yuv444=True)

b16 = fvf.Depth(dss, 16)

dbn = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
msk = kgf.retinex_edgemask(b16)
mrg = core.std.MaskedMerge(dbn, b16, msk)
snf = awf.rfs(b16, mrg, mappings="[135 191]")

dbn = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
msk = kgf.retinex_edgemask(b16).std.Expr("x 5000 > x 3 * x ?").std.Maximum().std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, b16, msk)
snf = awf.rfs(snf, mrg, mappings="[9927 9969] [39198 39252] [52377 52464] [67401 67449]")

lms = b16.std.ShufflePlanes(0, vs.GRAY).std.Expr("x 30784 = 65535 0 ?")
lmg = core.std.MaskedMerge(b16, mrg, lms)
snf = awf.rfs(snf, lmg, "[72423 80007]")

grn = adptvgrnMod(snf, size=1.5, sharp=90, strength=.7, cstrength=.4, static=False, luma_scaling=10)
snf = awf.rfs(grn, snf, "[0 299] [87024 88713]")

out = fvf.Depth(snf, 10)

out.set_output()