import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
from vsutil import *
import awsmfunc as awf
from rekt import *
from adptvgrnMod import *
import mvsfunc as mvf
import kagefunc as kgf

def fixbrdr(clip, thr=3):
	y = plane(clip, 0)
	diff_expr = "x y - abs"
	
	# prepare rows with enough size for convolutions
	row_1 = y.std.Crop(top=0, bottom=y.height - 1 - 3)
#	row_1_2 = y.std.Crop(top=0, bottom=y.height - 2)
	row_2 = y.std.Crop(top=1, bottom=y.height - 2 - 3)
#	row_2_3 = y.std.Crop(top=1, bottom=y.height - 3)
#	row_2_3_4 = y.std.Crop(top=2, bottom=y.height - 4)
	
	# FillMargins clip
	fill_mid = row_1.std.Convolution([0, 0, 0, 0, 0, 0, 2, 5, 2])
	fill_mid = fill_mid.std.CropAbs(width=y.width, height=1)
	
	# no idea what convolutions work best but these are the diagonal fills
	fill_left = row_1.std.Convolution([0, 0, 0, 0, 0, 0, 6, 2, 1])
	fill_left = fill_left.std.CropAbs(width=y.width, height=1)
	
	fill_right = row_1.std.Convolution([0, 0, 0, 0, 0, 0, 1, 2, 6])
	fill_right = fill_right.std.CropAbs(width=y.width, height=1)
	
	# need shifts to see if the row is diagonal or not
	shift_left = row_2.std.Convolution([0, 0, 0, 1, 0, 0, 0, 0, 0])
	shift_left = shift_left.std.CropAbs(width=y.width, height=1)
	shift_right = row_2.std.Convolution([0, 0, 0, 0, 0, 1, 0, 0, 0])
	shift_right = shift_right.std.CropAbs(width=y.width, height=1)
	
	# find threshold by looking at difference of third row minus its blur
	blur_left = row_2.std.Convolution([0, 0, 0, 1, 0, 0, 0, 0, 0])
	blur_left = blur_left.std.Convolution([0, 0, 0, 1, 0, 0, 0, 0, 0])
	blur_left = blur_left.std.Convolution([0, 1, 0, 1, 0, 1, 0, 1, 0])
	
	threshold_left = core.std.Expr([blur_left, row_2], diff_expr)
	threshold_left = threshold_left.std.CropAbs(width=y.width, height=1, top=1)
	
	blur_right = row_2.std.Convolution([0, 0, 0, 0, 0, 1, 0, 0, 0])
	blur_right = blur_right.std.Convolution([0, 0, 0, 0, 0, 1, 0, 0, 0])
	blur_right = blur_right.std.Convolution([0, 1, 0, 1, 0, 1, 0, 1, 0])
	
	threshold_right = core.std.Expr([blur_right, row_2], diff_expr)
	threshold_right = threshold_right.std.CropAbs(y.width, 1, top=1)
	
	# evaluate whether blur results in a pixel being too different from its shift
	sl_fm_diff = core.std.Expr([shift_left, fill_mid], diff_expr)
	sr_fm_diff = core.std.Expr([shift_right, fill_mid], diff_expr)
	#                    x           y           z               a                b          c           d
	fix = core.std.Expr([sl_fm_diff, sr_fm_diff, threshold_left, threshold_right, fill_right, fill_left, fill_mid], f"x z {thr} * > y a {thr} * < and b y {thr} a * > x z {thr} * < and c d ? ?")
	
	s = core.std.StackVertical([fix, y.std.Crop(top=1)])
	return core.std.ShufflePlanes([s, clip], [0, 1, 2], vs.YUV) if clip.format.color_family == vs.YUV else s


def fixbrdrs(clip, left=False, top=False, right=False, bottom=False, thr=3):
	if right:
		clip = fixbrdr(clip.std.FlipHorizontal().std.Transpose(), thr).std.FlipHorizontal().std.Transpose()
	if left:
		clip = fixbrdr(clip.std.Transpose(), thr).std.Transpose()
	if top:
		clip = fixbrdr(clip, thr)
	if bottom:
		clip = fixbrdr(clip.std.FlipVertical(), thr).std.FlipVertical()
	return clip
    

src = core.ffms2.Source("Parasite.2019.Black.and.White.Hybrid.BluRay.Remux.1080p.AVC.Atmos.7.1-HiFi.mkv").std.Crop(top=138, bottom=138)

# We'll need the UHD source to generate the mask for the black dots later.
uhd = core.ffms2.Source("parasite_deu_uhd.mkv").std.Crop(top=276, bottom=276)
uhd = depth(uhd, 16)
uhd = awf.cr(uhd, 1080, top=1, bottom=1)

src = depth(src, 16)

# No point in working on anything other than the luma plane here.
src = get_y(src)

############################################################################
# Dirty line fixes.  A simple rektlvls application wasn't entirely enough. #
############################################################################

# A diagonal line aware(-ish) version of FillBorders.
src = fixbrdrs(src, top=True, thr=1)

# Source has brights which extend past TV range, hence the extended prot_val.
src = rektlvls(src, rownum=[3, 4, 803, 802, 801, 800], rowval=[2, -3, 26, -10, 3, -3], prot_val=[16, 254])

# There are still some slight variable dirty lines, but bbmod's protection wasn't enough.
bbm = awf.bbmod(src, top=6, blur=20, thresh=2500, planes=0).std.Crop(bottom=src.height - 6)
csr = src.std.Crop(bottom=src.height - 6)
# A LimitFilter for brights, one for darks, and mix the two in between.
lmt_bright = mvf.LimitFilter(bbm, csr, thr=8, brighten_thr=5, elast=2)
lmt_dark = mvf.LimitFilter(bbm, csr, thr=1, brighten_thr=0.5, elast=1.2)
# This expression is honestly kind of dumb:
# If x is above bright/dark threshold, use the bright/dark limited clip.
# Else, softly apply all three clips.  Half of the source is used, the other half is made up of:
# dark limited * (1 - (x - 16) / 219) + bright limited * (x - 16) / 219, meaning for darker values, dark takes priority, else bright
# Is this ridiculous?  Yes.  Should the in-between equation have less priority on the source and just have been used for the whole clip?  Eh, maybe.  I'm writing this at 4 AM, so I don't really care anymore.
lmt = core.std.Expr([csr, lmt_bright, lmt_dark], f"x {50 << 8} < z  x {100 << 8} > y 1 x 4096 - 60160 4096 - / - z * x 4096 - 60160 4096 - / y * + 0.5 * x 0.5 * + ? ?")

# Super bright rows exist where dirty line fixes don't really work anymore.
csr = src.std.Crop(top=src.height - 8)
ref = src.cf.ContinuityFixer(bottom=8, radius=10).std.Crop(top=src.height - 8)
lmt_bot = mvf.LimitFilter(ref, csr, thr=9, brighten_thr=2, elast=1.1)
lmt_bot = core.std.Expr([csr, lmt_bot], f"x {210 << 8} > y x ?")

src = core.std.StackVertical([lmt, src.std.Crop(top=6, bottom=8), lmt_bot])

####################################################################
# There's a scene where there are random black dots flying around. #
####################################################################

# limit general area, although this should probably be animated later
ref = src.std.Crop(left=0, top=150, right=1650, bottom=524)
urf = uhd.std.Crop(left=0, top=150, right=1650, bottom=524)

# we need to ignore grain to build our mask
dn1 = core.knlm.KNLMeansCL(urf, d=0, s=3, h=1).resize.Bilinear(format=vs.YUV444P16)
dn2 = core.knlm.KNLMeansCL(get_y(urf).resize.Bilinear(urf.width / 2, urf.height / 2), d=1, s=5, h=1.5)

# the dots all have values within certain ranges
expr = "x {} < x {} > and 65535 0 ?"

# build mask to pick up dots
ms1 = dn1.std.Expr([expr.format(7350, 6750), "", expr.format(32550, 32950)])
# build mask on denoised and resized clip to find dot-free areas
ms2 = dn2.resize.Bilinear(urf.width, urf.height).std.Expr(expr.format(7200, 6600)).std.Maximum().std.Maximum()

# add the dot masks together and subtract dot-free mask
msk = core.std.Expr(split(ms1) + [ms2], "x z + a -").std.Maximum().std.Maximum().std.Maximum()

# shift pixels slightly so we don't get black edges then perform bilateral blur
# the idea is to blur the living shit out of the dots until they become brown again
def shift(clip):
    return clip.std.Convolution([2, 2, 1, 1, 0, 0, 1, 0, 0])
    
fix = iterate(ref, shift, 5)
fix = fix.bilateral.Bilateral(sigmaS=16, sigmaR=.01, algorithm=0)
# obviously have to grain after blurring
fix = adptvgrnMod(fix, luma_scaling=0, size=1.2, sharp=40, strength=.8, static=False, fade_edges=True, seed=7)

# merge via mask, add borders, merge via rekt
rep = core.std.MaskedMerge(ref, fix, msk).std.AddBorders(top=150, right=1650, bottom=524)
rep = rekt(src, rep, 0, 176, 1710, 524)
src = awf.rfs(src, rep, "[151784 151857]")

#######################################################################
# Debanding.  Source was relatively fine, so just light applications. #
#######################################################################

# Less writing for me to do later.
rtx = kgf.retinex_edgemask(src)
bms = awf.bandmask(src, thr=200, mid=2)

# Intro has an artifact that a simple blur makes unnoticeable.
blr = rekt_fast(src, lambda c: c.bilateral.Gaussian(sigma=10), 710, 650, 1130, 40)
blr = rekt(src, blr, 738, 680, 1148, 68)
src = awf.rfs(src, blr, "[165 169]")

# Intro requires some destructive debanding, but I prefer this over "details".
# It's just an intro, anyway, so who really cares?
dbn = src.placebo.Deband(threshold=6, radius=20, grain=0)
msk = kgf.kirsch(src).std.Inflate().std.Maximum()
dbn = dbn.std.MaskedMerge(src, msk)
grn = adptvgrnMod(dbn, size=1, strength=.16, fade_edges=True, seed=7, static=True, luma_scaling=2)
src = awf.rfs(src, grn, "[24 526]")

dbn = src.neo_f3kdb.Deband(y=32, grainy=8, sample_mode=4)
src = awf.rfs(src, dbn, "[552 572] [632 651]")

# Fades don't look great, sadly.
dbn = src.neo_f3kdb.Deband(range=30, y=64, grainy=0, sample_mode=4)
dbn = adptvgrnMod(dbn, size=1.1, strength=.1)
src = awf.rfs(src, dbn, "[1049 1059] [63630 63651] [166959 167047] [167112 167190] [181767 181917] [181971 182207] [182698 182737]")

# Banding in grainier frames.
dbn = src.neo_f3kdb.Deband(range=20, y=32, grainy=0, sample_mode=4)
ems = rtx.std.Binarize(12000).std.Maximum().std.Inflate()
dbn = dbn.std.MaskedMerge(src, ems)
dbn = src.std.MaskedMerge(dbn, bms)
dbn = adptvgrnMod(dbn, size=1.1, strength=.2, seed=20000, luma_scaling=20)
src = awf.rfs(src, dbn, "[5800 5885] [5976 6030] [8810 8891] [8965 9135] [9201 9377] [9442 9562] [43691 43796] [45866 46026] [58245 58377] [94847 94971] [98905 98946] [107066 107197] [119573 119634] [120202 120260] [126713 127867] [142331 143107] [144024 144507] [144508 145852] [157689 158159] [158246 158310] [158382 158463] [159764 159877] [164427 164434] [164548 164561] [171618 171691] [172301 172431] [175173 175509]")

# Bright skies often had banding or bad dither issues.
dbn = src.placebo.Deband(threshold=3, radius=30, grain=0)
# Absolutely no one cares if the grain is dynamic or not in brights.
dbn = sizedgrn(dbn, static=True, strength=.2)
msk = adptvgrnMod(src.std.Invert(), show_mask=True, luma_scaling=5)
dbn = src.std.MaskedMerge(dbn, msk)
src = awf.rfs(src, dbn, "[18631 18970] [54944 55133] [55301 55467] [55709 55794] [55884 56078] [56238 56381] [108917 108929] [149361 149413]")

# Frames with more prevalent banding issues.
dbn = src.neo_f3kdb.Deband(range=25, y=48, grainy=0, sample_mode=4)
ems = rtx.std.Expr("x 9000 > x 10 * x ?").std.Maximum().std.Inflate()
dbn = dbn.std.MaskedMerge(src, ems)
dbn = src.std.MaskedMerge(dbn, bms)
dbn = adptvgrnMod(dbn, luma_scaling=10, size=1.1, strength=.5, seed=2057)
src = awf.rfs(src, dbn, "[59873 59877] [143108 143327] [143702 144023] [159362 159365] [163396 163406] [171513 171617]")

# 4:2:0
src = src.resize.Point(format=vs.YUV420P8, dither_type="error_diffusion")

src.set_output()
