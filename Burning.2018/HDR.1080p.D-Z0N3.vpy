import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
from vsutil import *
import awsmfunc as awf
from adptvgrnMod import *
import kagefunc as kgf

src = core.ffms2.Source("Burning.2018.REPACK.Hybrid.UHD.BluRay.Remux.2160p.HEVC.HDR.DTS-HD.MA.5.1-eXcommunicado.mkv").std.Crop(top=276, bottom=276)
src_ = awf.zresize(src, 1080)
src = depth(src, 16)

src = awf.fb(src, top=1, bottom=1)

src = awf.fixlvls(src, min_in=18, max_in=224, min_out=16, max_out=235)

src = awf.zresize(src, 1080, top=1, bottom=1)

# fades
dbn = src.f3kdb.Deband(range=20, y=40, cb=40, cr=40, grainy=0, grainc=0)
y = get_y(src).std.Expr("x 65535 / sqrt sqrt 65535 *")
ems = kgf.kirsch(y).std.Binarize(1250).std.Maximum().std.Inflate().std.Maximum()
dbn = dbn.std.MaskedMerge(src, ems)
dbn = adptvgrnMod(dbn, size=1.1, strength=.2, cstrength=.1,  luma_scaling=50, lo=18)
src = awf.rfs(src, dbn, "[14779 14865] [193150 193435] [115999 116119] [208208 208364]")

# banded sky
bms = awf.bandmask(src, 75)
dbn = sizedgrn(dbn, strength=.1, cstrength=.05, static=True)
dbn = src.std.MaskedMerge(dbn, bms)
src = awf.rfs(src, dbn, "[115538 115985]")

src = depth(src, 10)

src.set_output()
