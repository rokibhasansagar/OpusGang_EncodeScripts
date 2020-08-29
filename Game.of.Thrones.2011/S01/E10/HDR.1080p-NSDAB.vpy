import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
from vsutil import *
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from debandmask import *

src = core.ffms2.Source("Game.of.Thrones.S01E10.Fire.and.Blood.2160p.UHD.BluRay.REMUX.HDR.HEVC.Atmos-EPSiLON.mkv")

src = fvf.Depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

db = src.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(src)
merged = core.std.MaskedMerge(db, src, mask)
sfdb = fvf.ReplaceFramesSimple(src, merged, mappings="[135 191] [75647 75666]")

db = src.f3kdb.Deband(range=15, y=25, cb=15, cr=15, grainy=48, grainc=15, output_depth=16, dynamic_grain=True)
mask = debandmask(src).std.Inflate()
merged = core.std.MaskedMerge(db, src, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings="[2629 2678] [8561 8621] [11022 11324] [13706 13737] [14147 14191] [14848 14982] [19291 19387] [22212 22675] [28525 28662] [28937 29019] [29140 29881] [30678 31156] [31198 31248] [31397 31884] [42314 42526] [42660 42808] [43435 43559] [71462 71530] [ ]")

final = fvf.Depth(sfdb, 10)

final.set_output()
