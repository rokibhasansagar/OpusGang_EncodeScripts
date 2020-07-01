import vapoursynth as vs
core = vs.core
import vsutil as vsu
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from rekt import *
from fixbrdrs import *
from adptvgrnMod import *
from functools import partial

src = core.ffms2.Source("The.World.Of.Kanako.2014.1080p.BluRay.REMUX.AVC.DTS-HD.MA5.1-MC.mkv").std.Crop(top=138, bottom=138)

src = vsu.depth(src, 16)

def evalf(n, f, clip):
    return rektlvls(clip, rownum=[0, 1, 2, 3, 4, 5, 797, 800, 801, 802], rowval=[13, -8, 6, -1, -1, 2, 2, 5, -9, 13], prot_val=[f.props.PlaneStatsMin >> 8 + 1, f.props.PlaneStatsMax >> 8 - 2])

src = src.std.PlaneStats()
src = src.std.FrameEval(partial(evalf, clip=src), src)
src = awf.fb(src, bottom=1, planes=[1, 2], top=1)
src = fixbrdrs(src, bottom=True)

src = src.knlm.KNLMeansCL(a=2, h=.14, d=2, device_type="gpu", channels="Y")
src = src.knlm.KNLMeansCL(a=2, h=.2, d=2, device_type="gpu", channels="UV")

# aliasing
# [30511 30630] [30864 30907]

dbn = fvf.gf3(src, thr=.5, elast=4, radius=25)
dbn = adptvgrnMod(dbn, luma_scaling=10, size=1, strength=.2, seed=222)
src = awf.rfs(src, dbn, "[24 407]")

dbn = src.f3kdb.Deband(range=20, y=72, cb=80, cr=80, grainy=0, grainc=0)
msk = kgf.retinex_edgemask(dbn).std.Expr("x 5000 > x 4 * x ?").std.Maximum().std.Inflate().std.Maximum()
msk.set_output()
dbn = dbn.std.MaskedMerge(src, dbn)
msk = awf.bandmask(src, thr=200)
dbn = src.std.MaskedMerge(dbn, msk)
dbn = adptvgrnMod(dbn, luma_scaling=40, size=1.5, strength=.8, cstrength=.5, sharp=80, seed=222)
src = awf.rfs(src, dbn, "[633 163056]")

src = vsu.depth(src, 10).std.Trim(23)

src.set_output()
