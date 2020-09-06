import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from rekt import *
from adptvgrnMod import *
from bandmask import *
from vsutil import *
core = vs.get_core()

#1 Import
src = core.ffms2.Source('/home/Videos/taste/remux.mkv')

#2 Crop change bit depth and chroma fix on left border
src = src.std.Crop(left=64, right=64)
src = fvf.Depth(src, 16)
src = awf.bbmod(src, left=2, thresh=None, blur=999, y=False, u=True, v=True)

#3 Deband #scene 1
dbn = core.f3kdb.Deband(src, range=15, y=56, cb=45, cr=40, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=150)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Binarize(8000).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=1.0, strength=.4, sharp=50, cstrength=.3, luma_scaling=40, seed=3)
snf = fvf.rfs(src, grn, "[133235 133682]")

#4 Deband #scene 2
dbn = core.f3kdb.Deband(src, range=15, y=56, cb=40, cr=30, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=150)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Binarize(8000).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=1.0, strength=.6, sharp=50, cstrength=.3, luma_scaling=5, seed=3)
snf = fvf.rfs(snf, grn, "[133683 134631]")

out = fvf.Depth(snf, 8)
out.set_output()
