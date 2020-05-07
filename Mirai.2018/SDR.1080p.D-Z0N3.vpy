import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import mvsfunc as mvf
import kagefunc as kgf
import vsTAAmbk as taa
import sgvsfunc as sgf
import havsfunc as haf
import blazefunc as bzf
import scoll
from debandmask import *
from rekt import *
from black_detect import *
from adptvgrnMod import *

#core.max_cache_size=20000

src = core.ffms2.Source("")

b16 = mvf.Depth(src, 16).std.Crop(top=20, bottom=22)

fb1 = core.fb.FillBorders(b16, top=1)
fb2 = core.fb.FillBorders(b16, top=2)
b16 = core.std.Merge(fb1, fb2, [0,1])

# Super light denoising.  No point in BM3D for denoising this weak.  Might end up changing to d=3, a=2 for both.
b16 = core.knlm.KNLMeansCL(b16, a=3, h=0.1, d=2, device_type='gpu', device_id=0, channels='Y')
b16 = core.knlm.KNLMeansCL(b16, a=2, h=0.2, d=1, device_type='gpu', device_id=0, channels='UV')

# Soft dehalo. 
b16 = haf.FineDehalo(b16, rx=2.0, thmi=80, thma=128, thlimi=25, thlima=100, darkstr=0.5, brightstr=0.5)

# Dirty lines during credits.
cf = core.std.Crop(b16, left=94, top=292, right=1018, bottom=290).fb.FillBorders(top=1, left=1, bottom=1).edgefixer.ContinuityFixer(left=[2, 2, 2], top=[2, 2, 2], right=2, bottom= [0, 2, 2], radius=15)
fb = rekt_fast(b16, lambda x: core.fb.FillBorders(x, left=2, top=2, right=1, bottom=2).std.Merge(cf, [1,0]), left=94, top=292, right=1018, bottom=290)
#fb = rektaa(fb, 94, 292, 1018, 290, mtype=2)
dh = rekt_fast(fb, lambda x: haf.FineDehalo(x, rx=2.0, thmi=80, thma=128, thlimi=25, thlima=100, darkstr=0.5, brightstr=2.3), left=94, top=292, right=1018, bottom=290)
sf = fvf.rfs(b16, dh, "[1434 2296]")

cf = core.std.Crop(b16, left=94, top=302, right=1018, bottom=300).fb.FillBorders(left=1).edgefixer.ContinuityFixer(left=[2, 2, 2], top=1, right=1, bottom= [1, 2, 2], radius=5)
fb = rekt_fast(b16, lambda x: core.fb.FillBorders(x, left=2, top=1, right=1, bottom=2).std.Merge(cf, [1,0]), left=94, top=302, right=1018, bottom=300)
dh = rekt_fast(fb, lambda x: haf.FineDehalo(x, rx=2.0, thmi=80, thma=128, thlimi=25, thlima=100, darkstr=0.5, brightstr=1.5), left=94, top=302, right=1018, bottom=300)
sf = fvf.rfs(sf, dh, "[133711 135117] [135360 136057] [136143 137216] [137282 138288] [138377 138757] [138820 140782]")

cf = core.std.Crop(b16, left=94, top=302, right=1018, bottom=300).fb.FillBorders(left=1).edgefixer.ContinuityFixer(left=[2, 2, 2], top=1, right=1, bottom= [1, 2, 2], radius=5)
fb = rekt_fast(b16, lambda x: core.fb.FillBorders(x, left=2, top=1, right=1, bottom=2).std.Merge(cf, [1,0]), left=94, top=302, right=1018, bottom=300)
dh = rekt_fast(fb, lambda x: haf.FineDehalo(x, rx=2.0, thmi=80, thma=128, thlimi=25, thlima=100, darkstr=0.5, brightstr=1.5).f3kdb.Deband(y=48, cb=0, cr=0, range=5, grainy=64, grainc=32, output_depth=16, keep_tv_range=True), left=94, top=302, right=1018, bottom=300)
sf = fvf.rfs(sf, dh, "[135118 135296] [138305 138376]")

mask = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Trim(2400, 2401) * src.num_frames
mask = rekt(mask, core.std.BlankClip(b16, 1920, 1038, format=vs.GRAY16), left=666, top=292, right=1114, bottom=744)
dh_lim = core.std.MaskedMerge(dh, b16, mask)
sf = fvf.rfs(sf, dh_lim, "[2297 2329]")

# 4:3 cropped scene.
crop = core.std.Crop(b16, left=254, right=254)
fb = core.fb.FillBorders(crop, left=1, right=1).std.Merge(crop, [1,0]).edgefixer.ContinuityFixer(left=1, right=1, top=0, bottom=0, radius=50).std.AddBorders(left=254, right=254, color=[4096, 32768, 32768])
sf = fvf.rfs(sf, fb, "[33448 34196]")

# Placebo edgemask binarized so we only get the obvious edges, then inflated.
mask = kgf.retinex_edgemask(b16).std.Binarize(65500).std.Maximum().std.Inflate()

# Strong aliasing.
aa = taa.TAAmbk(b16, aatype=2, mtype=0, opencl=False)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[4225 4727] [18340 18387] [129780 131148]")

# Mild aliasing.
aa = taa.TAAmbk(b16, aatype=3, mtype=0, opencl=False)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[55394 55451] [55649 55782] [120840 120901]")

# Very strong aliasing.
aa = taa.TAAmbk(b16, aatype=6, mtype=0, repair=16, opencl=False)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[ ] [107405 107462]")

# Strong aliasing that I tried to fix with a terrible mask.
mask = kgf.retinex_edgemask(b16).std.Binarize(65500).std.Maximum().std.Minimum(coordinates=[1,0,1,0,0,1,0,1]).std.Deflate().std.Deflate()
aa = taa.TAAmbk(b16, aatype=6, mtype=0, opencl=False)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[55510 55580]")

#########################
# REQUIRES MAJOR REPAIR #
#########################
#sf = fvf.rfs(sf, ?, "[65880 66478]")
#sf = fvf.rfs(sf, ?, "[120902 121051] [121790 121905] [122388 122528] [123038 123153] [126686 126812] [128740 128953]") #Banding? [121063 121095] [121906 121968] [122530 122576]

# Graining
gr = adptvgrnMod(b16, strength=2.5, size=1.25, sharp=35, static=False, luma_scaling=3, grain_chroma=False)
sf = fvf.rfs(sf, gr, "[120840 120901]")

# Debanding.
dbmask = debandmask(b16, lo=6144, hi=12288, lothr=320, hithr=384, mrad=2)
deband = core.f3kdb.Deband(b16, y=34, cb=0, cr=0, range=10, grainy=16, grainc=8, output_depth=16, keep_tv_range=True)
merge = core.std.MaskedMerge(deband, b16, dbmask)
merge = adptvgrnMod(merge, strength=2, size=1.5, sharp=25, static=False, luma_scaling=5, grain_chroma=True)
sf = fvf.rfs(sf, merge, "[3174 3254] [3540 3655] [7463 7749] [41056 41597] [63482 64106] [91033 91164]")
#[25 47]

# Debanding with retinex.
mask = kgf.retinex_edgemask(b16).std.Maximum().std.Inflate().std.Maximum().std.Inflate()
deband = core.f3kdb.Deband(b16, y=48, cb=48, cr=48, range=15, grainy=16, grainc=16, output_depth=16, keep_tv_range=True)
merge = core.std.MaskedMerge(deband, b16, mask)
merge = adptvgrnMod(merge, strength=2.2, size=1.25, sharp=15, static=False, luma_scaling=5, grain_chroma=True)
sf = fvf.rfs(sf, merge, "[77952 78034] [93358 93443]")

# Debanding with gradfun3 mask.
deband = Fag3kdb(b16, thry=54, thrc=54, radiusy=10, radiusc=6, grainy=32, grainc=16)
sf = fvf.rfs(sf, deband, "[25 263]")

final = mvf.Depth(sf, 8, dither=7)

blank = core.std.BlankClip(src.std.Crop(top=20, bottom=22), 1920, 1038, color=[16, 128, 128])
final = fvf.rfs(final, blank, "[0 24] [1352 1433] [58945 59016] [75563 75633] [78351 78421] [81130 81141] [81261 81272] [93967 94062] [99889 99959] [118093 118147] [140928 140951]")

final.set_output()

# zones
#655,703,crf=fade/2465,2512,crf=fade/12300,12347,crf=fade/75634,75669,crf=fade/78422,78467,crf=fade/135993,136057,crf=creditsfade/137921,138041,crf=creditsfade/138662,138757,crf=creditsfade/140688,140782,crf=creditsfade/