import vapoursynth as vs
core = vs.get_core()
import awsmfunc as awf
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
from adptvgrnMod import *
from vsutil import *
from rekt import rekt, rekt_fast
from bandmask import *

core.max_cache_size = 2048

src = core.ffms2.Source("/home/aicha/Videos/parasite.mkv")

src = fvf.Depth(src, 16)

src = src.std.Crop(top=276, bottom=276)

# single black row at top and bottom meaning chroma is also screwed there
src = awf.fb(src, top=1, bottom=1)
src = src.fb.FillBorders(top=2, bottom=2, mode="fillmargins").std.Merge(src, [1, 0])

# variable dirty lines on luma plane
src = awf.bbmod(src, top=3, bottom=3, thresh=5000, blur=25, u=False, v=False)

src = src.resize.Spline36(1280, 536, src_top=1, src_height=2160 - 2 * 276 - 2, format=vs.YUV420P16)

###################################################################
# there's a scene where there are random black dots flying around #
###################################################################

# limit general area
ref = src.std.Crop(left=0, top=100, right=1100, bottom=350)

# we need to ignore grain to build our mask
dn1 = core.knlm.KNLMeansCL(ref, d=0, s=3, h=1).resize.Bilinear(format=vs.YUV444P16)
dn2 = core.knlm.KNLMeansCL(get_y(ref).resize.Bilinear(ref.width / 2, ref.height / 2), d=1, s=5, h=1.5)

# the dots all have values within certain ranges
expr = "x {} < x {} > and 65535 0 ?"

# build mask to pick up dots
ms1 = dn1.std.Expr([expr.format(7350, 6750), "", expr.format(32550, 32950)])
# build mask on denoised and resized clip to find dot-free areas
ms2 = dn2.resize.Bilinear(ref.width, ref.height).std.Expr(expr.format(7200, 6600)).std.Maximum().std.Maximum()

# add the dot masks together and subtract dot-free mask
msk = core.std.Expr(split(ms1) + [ms2], "x z + a -").std.Maximum().std.Maximum().std.Maximum()

# shift pixels slightly so we don't get black edges then perform bilateral blur
# the idea is to blur the living shit out of the dots until they become brown again
def shift(clip):
    return clip.std.Convolution([2, 2, 1, 1, 0, 0, 1, 0, 0])
    
fix = iterate(ref, shift, 5)
fix = fix.bilateral.Bilateral(sigmaS=14, sigmaR=.01, algorithm=0)
# obviously have to grain after blurring
fix = adptvgrnMod(fix, luma_scaling=0, size=1.2, sharp=40, strength=.3, cstrength=.3, static=False, fade_edges=True, seed=7)

# merge via mask, add borders, merge via rekt
rep = core.std.MaskedMerge(ref, fix, msk).std.AddBorders(top=100, right=1100, bottom=350)
rep = rekt(src, rep, 0, 116, 1140, 348)
snf = fvf.rfs(src, rep, "[151784 151857]")

###############################################################################
# x265 smoothing is atrocious, so we're graining B and P frames significantly #
###############################################################################

# seed is set for the sake of comparisons
snf = frmtpgrn(snf, size=1.1, strength=[.3, .15, .05], cstrength=[.15, .1, .05], sharp=3 * [75], luma_scaling=10, grain_chroma=True, fade_edges=True, static=False, show_mask=False, seed=7)

#############
# debanding #
#############

dbn = core.f3kdb.Deband(src, range=15, y=32, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=150)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Binarize(10000).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=1.1, strength=.4, cstrength=.15, sharp=75, static=False, fade_edges=True, luma_scaling=25, seed=7)
snf = fvf.rfs(snf, grn, "[18771 19202] [42268 42808] [42809 43981] [54854 56078] [66676 71508] [77260 77532] [107066 107197] [107578 107772] [119573 119634] [120202 120260] [125308 125380] [131573 131693] [140814 140906] [142092 142154] [142171 145852] [157689 158463] [159215 159323] [159347 159489] [159764 159877] [159968 160002] [160152 160269] [164424 164446] [166970 167049] [167112 167160] [171513 171617] [172301 172431] [174036 174223] [175173 175509] [175606 175690] [176501 176606] [181769 181917] [181967 182134]")

# deband intro so we don't have to cut it
# I just threw a huge deband over it because whatever it's an intro who really gives a damn
dbn = fvf.gf3(src, thr=1, radius=15, elast=7, mask=1)
dbg = adptvgrnMod(dbn, size=1, grain_chroma=False, strength=.1, fade_edges=True, seed=7)
snf = fvf.rfs(snf, dbg, "[24 527]")

# intro has an artifact that a simple blur makes unnoticeable
blr = rekt_fast(dbg, lambda c: c.bilateral.Gaussian(sigma=10), 472, 432, 752, 26)
blr = adptvgrnMod(blr, size=1, grain_chroma=False, strength=.1, fade_edges=True, static=True, seed=7)
blr = rekt(dbg, blr, 492, 452, 764, 42)
snf = fvf.rfs(snf, blr, "[165 169]")

# black outro
blk = src.std.BlankClip(color=[4096, 32768, 32768])
msk = get_y(src).std.Binarize(4097)
mrg = blk.std.MaskedMerge(src, msk)
snf = fvf.rfs(snf, mrg, "[0 23] [182739 189742]")

out = fvf.Depth(snf, 8)

out.set_output()
