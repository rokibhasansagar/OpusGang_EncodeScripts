import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
from vsutil import *
import awsmfunc as awf
from rekt import *
import kagefunc as kgf
from adptvgrnMod import *

src = core.ffms2.Source("Wasp.Network.2019.1080p.Blu-ray.Remux.AVC.DTS-HD.MA 5.1-WASP.mkv").std.Crop(top=140, bottom=140)

src = depth(src, 16)

############################################
# Fill dirty chroma and adjust dirty luma. #
############################################

src = awf.fb(src, top=1, planes=[1, 2])

src = rektlvls(src, rownum=[0, 1], rowval=[28, -1.5], prot_val=[16, 235])

##################################################
# Crop borders, fix dirty lines, re-add borders. #
##################################################

def readdborder(clip, left=0, right=0, top=0, bottom=0, lsat=.88, tsat=.2, rsat=None, bsat=.2, fb=False, bbmod=False):
    fix = src.std.Crop(left, right, top, bottom)
    if fb:
        fix = awf.fb(fix, *fb, planes=[0, 1, 2])
    if bbmod:
        fix = awf.bbmod(fix, left=bbmod[0], right=bbmod[1], top=bbmod[2], bottom=bbmod[3], thresh=bbmod[4], blur=bbmod[5], scale_thresh=True, u=bbmod[6], v=bbmod[6])
    return awf.AddBordersMod(fix, left=left, right=right, top=top, bottom=bottom, lsat=1, tsat=.2, rsat=None, bsat=.2, color=[4096, 32768, 32768])

fix = readdborder(src, left=196, top=6, right=192, bottom=6, lsat=1, tsat=1, rsat=None, bsat=1, fb=[0, 0, 1, 0])
src = awf.rfs(src, fix, "[102358 102587]")

fix = readdborder(src, left=226, top=6, right=212, bottom=6, lsat=1, tsat=1, rsat=.7, bsat=1, fb=[0, 1, 1, 0], bbmod=[6, 6, 0, 0, 32, 30, False])
src = awf.rfs(src, fix, "[102588 103586]")

fix = readdborder(src, left=408, top=0, right=402, bottom=0, lsat=1, tsat=1, rsat=1, bsat=1, bbmod=[1, 1, 0, 0, 50, 20, True])
src = awf.rfs(src, fix, "[111767 112169]")

fix = readdborder(src, left=408, top=4, right=402, bottom=6, lsat=1, tsat=1, rsat=1, bsat=1, bbmod=[0, 1, 0, 0, 50, 500, True])
src = awf.rfs(src, fix, "[163669 165509]")

fix = readdborder(src, left=10, bbmod=[1, 0, 0, 0, 70, 999, False])
src = awf.rfs(src, fix, "86648")

#############################
# Similar thing for splits. #
#############################

def blksplit(clip, left, right):
    lft = clip.std.Crop(left=left)
    lft = awf.bbmod(lft, left=2)
    rit = clip.std.Crop(right=right)
    rit = awf.bbmod(rit, right=2)
    blk = clip.std.BlankClip(width=clip.width - lft.width - rit.width, color=[4096, 32768, 32768])
    return core.std.StackHorizontal([rit, blk, lft])
fix = blksplit(src, 598, 1336)
src = awf.rfs(src, fix, "[86733 86923]")

fix = blksplit(src, 1362, 570)
src = awf.rfs(src, fix, "[86855 86927]")

#############################################
# And for pictures inserted during credits. #
#############################################

def credits(y, u, v, left, top, right, bottom):
    y = rekt_fast(y, lambda x: x.cf.ContinuityFixer(1, 1, 1, 1, radius=5), left, top, right, bottom)
    fix = join([y, u, v])
    ml = left % 2
    mt = top % 2
    mr = right % 2
    mb = bottom % 2
    return rekt_fast(fix, lambda x: awf.fb(x, left=ml, top=mt, right=mr, bottom=mb, planes=[1, 2]), left + ml, top + mt, right + mr, bottom + mb)

y, u, v = split(src)

fix = credits(y, u, v, 161, 133, 1263, 116)
src = awf.rfs(src, fix, "[174895 174991]")

fix = credits(y, u, v, 162, 131, 1264, 115)
src = awf.rfs(src, fix, "[175003 175088]")

fix = credits(y, u, v, 1222, 130, 206, 123)
src = awf.rfs(src, fix, "[175098 175193]")

fix = credits(y, u, v, 1221, 132, 205, 121)
src = awf.rfs(src, fix, "[175206 175331]")

fix = credits(y, u, v, 131, 119, 1213, 118)
src = awf.rfs(src, fix, "[175350 175429]")

fix = credits(y, u, v, 131, 120, 1165, 119)
src = awf.rfs(src, fix, "[175442 175605]")

fix = credits(y, u, v, 1221, 129, 204, 122)
src = awf.rfs(src, fix, "[175615 175706]")

fix = credits(y, u, v, 1217, 128, 204, 122)
src = awf.rfs(src, fix, "[175719 175818]")

fix = credits(y, u, v, 162, 133, 1264, 116)
src = awf.rfs(src, fix, "[175824 175878]")

fix = credits(y, u, v, 162, 134, 1263, 119)
src = awf.rfs(src, fix, "[175891 175955]")

fix = credits(y, u, v, 1221, 130, 206, 124)
src = awf.rfs(src, fix, "[175965 176039]")

fix = credits(y, u, v, 1221, 130, 206, 125)
src = awf.rfs(src, fix, "[176052 176143]")

##########################################################
# Whole movie needs debanding, so I did some dumb stuff. #
##########################################################

# Deband with libplacebo, split for performance reasons.
y, u, v = split(src)
y = y.placebo.Deband(threshold=6, radius=26.0, grain=0)
u = u.placebo.Deband(threshold=3, radius=16.0, grain=0)
v = v.placebo.Deband(threshold=3, radius=16.0, grain=0)
dbn = join([y, u, v])
# Grain with static grain, as it's less noticeable in brights and darks come later. 
dbn = sizedgrn(dbn, strength=.1, size=1.2, sharp=40, static=True, seed=1991)
# Protect details using a strong edge mask. 
ems = kgf.retinex_edgemask(src).std.Expr("x 7500 > x 3 * x 0.8 * ?").std.Inflate()
# Limit all this to very smooth areas.
bms = awf.bandmask(src, thr=100, mid=2)
dbn = dbn.std.MaskedMerge(src, ems)
src = awf.rfs(src, dbn, "[105 372] [405 929]")
dbn = src.std.MaskedMerge(dbn, bms)
# Add dynamic grain to darks. 
dbn = adptvgrnMod(dbn, strength=.2, cstrength=.1, size=1.3, sharp=40, static=False, luma_scaling=50, seed=1990)
src = awf.rfs(src, dbn, "[1781 174690]")

src = depth(src, 8)

src.set_output()
