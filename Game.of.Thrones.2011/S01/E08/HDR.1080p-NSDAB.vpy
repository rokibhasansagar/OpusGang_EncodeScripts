import vapoursynth as vs
core = vs.core
import fvsfunc as fvf
import kagefunc as kgf
import awsmfunc as awf
from vsutil import *
from adptvgrnMod import *

core.max_cache_size = 1024

src = core.ffms2.Source("Game.of.Thrones.S01E08.The.Pointy.End.2160p.UHD.BluRay.REMUX.HDR.HEVC.Atmos-EPSiLON.mkv")

src = depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

dbn = src.neo_f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=0, grainc=0, output_depth=16, sample_mode=4)
dbn = sizedgrn(dbn, strength=.1, cstrength=.05, static=True)
msk = kgf.retinex_edgemask(src)
mrg = core.std.MaskedMerge(dbn, src, msk)
src = awf.rfs(src, mrg, "[135 191]")

dbn = src.neo_f3kdb.Deband(range=20, y=48, cb=48, cr=48, grainy=0, grainc=0, output_depth=16, sample_mode=4)
dbn = sizedgrn(dbn, strength=.7, cstrength=.4, sharp=90, size=1.5)
msk = msk.std.Binarize(8000).std.Maximum()
dbn = dbn.std.MaskedMerge(src, msk)
bms = awf.bandmask(src, thr=100).std.Inflate().std.Maximum().std.Inflate()
dbn = src.std.MaskedMerge(dbn, bms)
agm = adptvgrnMod(src, show_mask=True, luma_scaling=10)
dbn = src.std.MaskedMerge(dbn, agm)
src = awf.rfs(src, dbn, "[226 81988]")

src = depth(src, 10)
