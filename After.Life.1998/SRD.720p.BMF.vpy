import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
import vsutil as vsu
import havsfunc as haf
from rekt import *
from adptvgrnMod import *
from bandmask import *
from autodbnd import autodbnd
core = vs.get_core()

#1 Import / crop / bit depth

src = core.lsmas.LWLibavSource('After.Life.1998.BluRay.Remux.1080p.AVC.FLAC.2.0 BMF.mkv')
src = src.std.Crop(left=64, right=64)
src = depth(src, 16)
src = awf.zresize(src, preset=720)

#3 Debanding

dbn = core.f3kdb.Deband(src, range=15, y=58, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=200)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Expr("x 9500 > x 4 * x ?").std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=1.4, strength=0.15, sharp=80, cstrength=None, luma_scaling=100, seed=3)
snf = awf.rfs(src, grn, "[427 473][104435 104777][160880 160915]")


out = depth(snf, 8)
out.set_output()

