import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
from vsutil import get_y
from adptvgrnMod import *
import kagefunc as kgf
import awsmfunc as awf
import havsfunc as haf
from sizedgrn import *

src = core.ffms2.Source("video.mkv").std.Crop(top=138, bottom=138)

s16 = fvf.Depth(src, 16)
y = get_y(s16)

# Debanding the intro.  Muh autistic GF3 use.
lumamask = y.std.Binarize(25000)
deband = fvf.GradFun3(s16, thr=1, mask=0, elast=5, radius=10)
deband = fvf.GradFun3(deband, thr=1, mask=0, elast=10, radius=30).grain.Add(constant=True, var=.15)
merge = core.std.MaskedMerge(deband, s16, lumamask)
sf = fvf.rfs(s16, merge, "[50 133]")

# Ringing during intro credits.
mask = y.std.Sobel().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Binarize(60000)
rmask = core.std.BlankClip(y, color=65535)
lmask = y.std.Binarize(20000).std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Invert()
dering = haf.HQDeringmod(s16, drrep=24, thr=20, elast=3, sharp=0, ringmask=rmask)
dering = sizedgrn(dering, size=1.5, sharp=33, strength=3, static=False)
merge = core.std.MaskedMerge(s16, dering, mask)
merge = core.std.MaskedMerge(merge, s16, lmask)
sf = fvf.rfs(sf, merge, "[792 3845]") # "[1278 1391] [1497 1607] [1717 1821] [1928 2010] [2123 2200] [2316 2398] [2510 2593] [2705 2789] [2893 2978] [3093 3172] [3286 3370]")

# These three are to fix some banding that occurred during a few scenes.
mask = kgf.retinex_edgemask(s16).std.Binarize(5000).std.Inflate()
deband = core.f3kdb.Deband(s16, range=10, y=64, cr=32, cb=32, grainy=32, grainc=32, dynamic_grain=False, output_depth=16)
merge = core.std.MaskedMerge(deband, s16, mask)
grain = adptvgrnMod(merge, size=1.8, sharp=33, luma_scaling=13, static=False)
sf = fvf.rfs(sf, grain, "[15108 15218] [15234 15264] [107815 109153] [193511 193674] [193691 194169]")

grain = adptvgrnMod(merge, strength=0.1, size=1.8, sharp=33, luma_scaling=13, static=False)
sf = fvf.rfs(sf, grain, "[115923 116370] [116391 116516]")

grain = adptvgrnMod(merge, strength=0.3, size=1.6, sharp=90, luma_scaling=5, grain_chroma=True, static=False)
sf = fvf.rfs(sf, grain, "[208506 208746]")

# Banding in a dark area.
lumamask = y.std.Binarize(6150).std.Invert()
grain = adptvgrnMod(merge, size=2, sharp=33, luma_scaling=5, static=False)
merge = core.std.MaskedMerge(merge, grain, lumamask)
sf = fvf.rfs(sf, merge, "[111145 111388]")

# Chroma blocking during a few scenes.  Also adding some luma grain for good measure.
deblock_chroma = core.dfttest.DFTTest(s16, sigma=5, planes=[1])
deblock_chroma = adptvgrnMod(deblock_chroma, size=1.9, sharp=50, luma_scaling=30, static=False, grain_chroma=False)
sf = fvf.rfs(sf, deblock_chroma, "[93226 93810] [111991 113487]")

# Fixing some dirty lines on the left and right in bright areas.
lvl = awf.bbmod(s16, left=2, right=2)
left = core.std.Crop(y, right=1916)
right = core.std.Crop(y, left=1916)
leftmask = left.std.Binarize(30000).std.Inflate().std.Inflate()
rightmask = right.std.Binarize(30000).std.Inflate().std.Inflate()
blank = core.std.BlankClip(y, 1912, 804)
mask = core.std.StackHorizontal([leftmask, blank, rightmask])
sf = core.std.MaskedMerge(sf, lvl, mask)

# Scene in which HD BD has too much ringing compared to UHD
uhd = core.ffms2.Source("uhd.mkv").std.Crop(top=276, bottom=276)
uhd = fvf.Depth(uhd, 16)
uhd = awf.DynamicTonemap(uhd)
fmg = core.std.Merge(awf.fb(uhd, bottom=1, top=1), awf.fb(uhd, bottom=2, top=2), [0, 1])
bbm = awf.bbmod(fmg, bottom=2)
uhd = uhd.resize.Spline36(1920, 804, src_top=1, src_height=1606, dither_type="error_diffusion", format=vs.YUV444P16)
ups = uhd.resize.Point(format=vs.RGB48)
lut = ups.timecube.Cube("2.cube")
uhd = lut.resize.Spline36(format=vs.YUV420P16, matrix_s="709")
uhd = s16.std.Trim(0, 384) + uhd 
uhd = adptvgrnMod(uhd, strength=.7, size=2, sharp=33, luma_scaling=3, static=False, grain_chroma=True)
sf = fvf.rfs(sf, uhd, "[117237 117573]")

lut = ups.timecube.Cube("1.cube")
uhd = lut.resize.Spline36(format=vs.YUV420P16, matrix_s="709")
uhd = s16.std.Trim(0, 384) + uhd 
uhd = adptvgrnMod(uhd, strength=.7, size=2, sharp=33, luma_scaling=3, static=False, grain_chroma=True)
sf = fvf.rfs(sf, uhd, "[116859 117049]")

# Dithering down.
final = fvf.Depth(sf, 8)

final.set_output()