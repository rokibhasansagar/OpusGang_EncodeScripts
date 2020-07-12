import vapoursynth as vs
core = vs.core
from vsutil import *
import awsmfunc as awf
from fixbrdrs import *
from rekt import *
import fvsfunc as fvf
import kagefunc as kgf
from adptvgrnMod import *

src = core.ffms2.Source("antiporno_fra.mkv").std.Crop(top=20, bottom=20)

src = depth(src, 16)

###############################
# source has normal gamma bug #
###############################

src = awf.fixlvls(src, gamma=.88)

########################################################################################################
# A rather complicated fix for some darkened pixels in the second to last row every now and then.      #
# When the error-free row above the affected row minus the affected row is greater than the threshold, #
# replace the pixel with FillBorders, else leave pixel.  Limit this to pixels in which the difference  #
# between FillBorders and the source row is above a threshold and the reference pixel is above a       #
# luma threshold.  An example frame is 5116.                                                           #
########################################################################################################

y, u, v = split(src)
y1 = y.std.Crop(top=y.height - 2, bottom=1)
y2 = y.std.Crop(top=y.height - 3, bottom=2)
yf = y.std.Crop(top=y.height - 3, bottom=1).fb.FillBorders(bottom=1, mode="fillmargins").std.Crop(top=1)
flb = core.std.Expr([y1, y2, yf], "x y - 100 > x z ?")
flb = core.std.StackVertical([y.std.Crop(bottom=2), flb, y.std.Crop(top=y.height - 1)])
flb = join([flb, u, v])
ref = src.std.Crop(bottom=2).std.AddBorders(top=2)

src = core.std.Expr([src, flb, ref], ["x y - abs 100 > z 7000 > and y x ?", ""])

########################################
# Luma and chroma fill top and bottom. #
########################################

src = awf.fb(src, top=1, bottom=1, planes=[1, 2])
src = fixbrdrs(src, top=True, bottom=True)

#########################################################################################################
# Dirty lines in intro, bottom left corner required a fill limited to areas where difference was large. #
#########################################################################################################

fix = src.cf.ContinuityFixer(left=[3, 0], top=[2, 0], right=[1, 0], radius=10)
fix = awf.fb(fix, left=1, planes=[1, 2])
def partfix(c):
    fix = awf.fb(c, left=5)
    return core.std.Expr([c, fix], "x y - abs 200 > y x ?")
fix = rekt_fast(fix, partfix , 0, 1026, 1912)
src = awf.rfs(src, fix, "[2 431]")

src = src.resize.Spline36(1280, 694, src_top=1, src_height=src.height - 2)

ret = kgf.retinex_edgemask(src)
y = get_y(src)

###############################################
# Intro deband was easiest with a strong gf3. #
###############################################

dbn = fvf.gf3(src, thr=1, elast=4, mask=2)
dbn = adptvgrnMod(dbn, strength=.2, static=True, luma_scaling=3)
src = awf.rfs(src, dbn, "[2 431]")

#############################################################################################################
# A bunch of debands with slightly different parameters.  Most of them are just limited to grain-free areas #
# via bandmask with details protected by a strong retinex_edgemask, and finally grained via adptvgrnMod.    #
#############################################################################################################

dbn = src.neo_f3kdb.Deband(range=15, y=80, cb=72, cr=72, grainy=0, grainc=0, sample_mode=4, output_depth=16)
msk = y.std.Binarize(140 << 8)
dbn = src.std.MaskedMerge(dbn, msk)
dbn = sizedgrn(dbn, strength=.3, cstrength=.15, static=True)
msk = awf.bandmask(src, 150)
dbn = src.std.MaskedMerge(dbn, msk)
src = awf.rfs(src, dbn, "[3655 3997] [4533 5096] [94211 94715]")

dbn = src.neo_f3kdb.Deband(range=15, y=56, cb=56, cr=56, grainy=0, grainc=0, sample_mode=4, output_depth=16)
dbn = src.std.MaskedMerge(dbn, msk)
ems = ret.std.Binarize(10000).std.Maximum()
dbn = dbn.std.MaskedMerge(src, ems)
dbn = adptvgrnMod(dbn, strength=.6, cstrength=.4, size=1.5, sharp=60, static=False, luma_scaling=10)
src = awf.rfs(src, dbn, "[5661 5718] [59710 59715] [92211 92264] [5097 5193] [5890 6229] [59248 59483] [59680 59709] [87600 87714] [39852 39853] [92671 92789] [54248 54313]")

# This part was black and white.
dbn = dbn.std.Merge(dbn.std.BlankClip(), [0, 1])
src = awf.rfs(src, dbn, "[53251 54029]")

# Gradients looked bad, so applied a strong gf3.
dbn = fvf.gf3(src, thr=1, elast=3, mask=0)
msk = awf.bandmask(src, 300, mid=2)
dbn = src.std.MaskedMerge(dbn, msk)
ems = ret.std.Expr("x 13000 > x 5 * x ?").std.Maximum().std.Inflate().std.Inflate()
dbn = dbn.std.MaskedMerge(src, ems)
dbn = adptvgrnMod(dbn, strength=.6, cstrength=.4, size=1.5, sharp=60, static=False, luma_scaling=10)
src = awf.rfs(src, dbn, "[52522 52669]")

dbn = src.neo_f3kdb.Deband(range=15, y=56, cb=56, cr=56, grainy=64, grainc=48, sample_mode=4, output_depth=16)
dbn = src.std.MaskedMerge(dbn, msk)
src = awf.rfs(src, dbn, "[101990 102072]")

dbn = src.neo_f3kdb.Deband(range=30, y=64, cb=64, cr=64, grainy=32, grainc=32, sample_mode=4, output_depth=16)
src = awf.rfs(src, dbn, "[104916 106080]")

##################################################################################################################
# There's a bunch of ringing here, which becomes almost unnoticeable after downscaling.  Some blocking persists, #
# but it's so weak that deblocking doesn't make sense, hence I grained instead.                                  #
##################################################################################################################

dbn = adptvgrnMod(src, strength=.8, cstrength=.4, size=1.5, sharp=60, static=True, luma_scaling=30)
src = awf.rfs(src, dbn, "[100807 101003] [101048 101255] [101326 101401] [101478 101546] [101643 101789] [101915 101989] [102213 102311] [102403 102549] [102677 102800] [102935 103191] [103447 103557] [103611 103725]")

####################################################################################################################
# There are a lot of reds.  Grained very red areas with static grain in an attempt to allocate more bits to these. #
####################################################################################################################

rgb = src.resize.Bilinear(format=vs.RGB48, matrix_in_s="709")
rgb = core.std.Expr(split(rgb), f"x 1.5 * y - z - x 2 * y - min")
rgb = rgb.std.Invert().std.PlaneStats().adg.Mask(20)
grn = sizedgrn(src, strength=.3, cstrength=.1, size=1.5, sharp=60, static=True)
src = src.std.MaskedMerge(grn, rgb)

src = depth(src, 8)

src.set_output()
