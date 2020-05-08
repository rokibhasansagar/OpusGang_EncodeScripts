import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as haf
from fag3kdb import *
from debandmask import *
from adptvgrnMod import *

src = core.ffms2.Source("").std.Crop(top=138, bottom=138)

s16 = fvf.Depth(src, 16)

ref = core.knlm.KNLMeansCL(s16, a=3, h=0.3, d=1, device_type='gpu', device_id=0, channels='Y')
retinex = kgf.retinex_edgemask(ref)

db = haf.Deblock_QED(s16, quant1=30, quant2=32)
deband = core.f3kdb.Deband(db, range=5, y=88, cr=0, cb=0, grainy=48, grainc=32, dynamic_grain=False, output_depth=16)
merge = core.std.MaskedMerge(deband, s16, retinex)
sf = fvf.rfs(s16, deband, "[133 382]")

dbmask = retinex.std.Binarize(5000).std.Inflate()
y = core.knlm.KNLMeansCL(s16, a=4, h=0.4, d=2, device_type='gpu', device_id=0, channels='Y')
uv = core.knlm.KNLMeansCL(s16, a=3, h=0.5, d=1, device_type='gpu', device_id=0, channels='UV')
prep = core.std.ShufflePlanes([y, uv], [0, 1, 2], vs.YUV)
deband = core.f3kdb.Deband(prep, range=20, y=48, cr=48, cb=48, grainy=32, grainc=16, dynamic_grain=True, output_depth=16)
merge = core.std.MaskedMerge(deband, s16, dbmask)
merge = adptvgrnMod(sf, strength=.6, size=.8, sharp=70, static=False, luma_scaling=11, grain_chroma=True)

dbmask = retinex.std.Binarize(25000).std.Minimum(coordinates=[0, 2, 0, 0, 0, 0, 2, 0]).std.Inflate()
deband = core.f3kdb.Deband(s16, range=15, y=80, cr=0, cb=0, grainy=32, grainc=16, dynamic_grain=True, output_depth=16)
merge = core.std.MaskedMerge(deband, s16, dbmask)
merge = adptvgrnMod(merge, strength=.6, size=.8, sharp=70, static=False, luma_scaling=11, grain_chroma=True)
sf = fvf.rfs(sf, merge, "[1871 1944] [1973 1995] [144401 144491]")

final = fvf.Depth(merge, 10)

black = core.std.BlankClip(final, format=vs.YUV420P10, color=[64, 512, 512])
final = fvf.rfs(final, black, "[144492 144567] 149006")

mask = get_y(fvf.Depth(src, 10)).std.Binarize(80).std.Maximum().std.Inflate()
merge = core.std.MaskedMerge(black, fvf.Depth(src, 10), mask)
final = fvf.rfs(final, merge, "[144568 149005]")

final.set_output()