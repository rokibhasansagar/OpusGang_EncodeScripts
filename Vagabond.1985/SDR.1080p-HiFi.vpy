import vapoursynth as vs
import awsmfunc as awf
import vsutil
import kagefunc as kgf
from adptvgrnMod import *
 
core = vs.core
 
src = core.ffms2.Source('Vagabond.1985.1080p.Blu-ray.Remux.AVC.FLAC.1.0-CRITERION.mkv')
 
crp = src.std.Crop(left=64, right=64)
 
b16 = vsutil.depth(crp, 16)
 
msk = core.std.ShufflePlanes(b16, 0, vs.GRAY)
msk = msk.std.Binarize(8000).std.Invert()
grn = adptvgrnMod(b16, strength=9/6, size=3/2, sharp=45)
grn = core.std.MaskedMerge(b16, grn, msk)
 
out = vsutil.depth(grn, 8)
 
out.set_output()
