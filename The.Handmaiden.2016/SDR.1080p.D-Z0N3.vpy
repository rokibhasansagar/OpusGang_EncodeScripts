import vapoursynth as vs
core = vs.core
core.max_cache_size = 1024
from vsutil import *
import awsmfunc as awf
import kagefunc as kgf
import havsfunc as haf
from adptvgrnMod import *
from rekt import *
from retinex_fdog import *

kor = core.ffms2.Source("The Handmaiden 2016 1080p KOR.mkv").std.Crop(top=138, bottom=138)

src = core.ffms2.Source("The Handmaiden 2016 1080p JPN.mkv").std.AssumeFPS(kor)
src = src.std.Crop(top=138, bottom=138)
src_= src
src = depth(src, 16)

###############
# Dirty lines #
###############

src = rektlvls(src, colnum=[0, src.width - 1], colval=[5, 5], rownum=[0, src.height - 1], rowval=[5, 4.5], prot_val=[16, 240])

#############
# Debanding #
#############

dbn = src.placebo.Deband(threshold=4.5, radius=20, grain=0)
msk = kgf.kirsch(get_y(src)).std.Expr("x 5000 > x 4 * x ?").std.Inflate().std.Maximum()
dbn = dbn.std.MaskedMerge(src, msk)
grn = adptvgrnMod(dbn, size=1, strength=.16, fade_edges=True, seed=7, static=True, luma_scaling=2)
src = awf.rfs(src, grn, "[0 480]")

dbn = src.neo_f3kdb.Deband(range=25, y=72, cb=0, cr=0, grainy=0, grainc=0, sample_mode=4)
dbn = adptvgrnMod(dbn, strength=.2, static=True, luma_scaling=100, seed=4235)
src = awf.rfs(src, dbn, "[481 1009]")

def m2(v):
    return v + v % 2

def pointgrn(clip, size=1/2.3, strength=.25, cstrength=.3, static=False, seed=8493):
    dwn = clip.resize.Point(m2(round(clip.width * size)), m2(round(clip.height * size)))
    grn = dwn.grain.Add(var=strength, uvar=cstrength, constant=static, seed=seed)
    return grn.resize.Point(clip.width, clip.height)

dbn = src.neo_f3kdb.Deband(range=20, y=56, cb=32, cr=32, grainy=0, grainc=0, sample_mode=4)
msk = retinex_fdog(src, 2).std.Maximum().std.Inflate().std.Expr("x 15000 > x 2 * x .8 * ?").std.Inflate()
dbn = dbn.std.MaskedMerge(src, msk)
grn = adptvgrnMod(dbn, size=1, luma_scaling=70, grainer=pointgrn)
src = awf.rfs(src, grn, "[26756 26819] [27245 27279] [41638 41736] [57500 57532] [73439 73650] [76055 76206] [81724 81746] [82239 82261] [82466 82593] [91706 93141] [93327 93601] [93692 93990] [124155 124210] [127212 127354] [128369 128578] [128732 129252] [132018 132331] [139179 139280] [169474 169692] [183287 183462] [194932 194945] [194993 195006] [195032 195053] [195116 195135] [195161 195183] [195539 195825] [196020 196114]")

bms = awf.bandmask(src, thr=100)
mrg = src.std.MaskedMerge(dbn, bms)
gms = adptvgrnMod(dbn, show_mask=True).std.Invert()
grn = sizedgrn(mrg, static=True)
grn = src.std.MaskedMerge(grn, gms)
src = awf.rfs(src, grn, "[76912 77571] [85021 85041] [168490 168499] [195407 195538] [195826 196019]")

#############
# Deringing #
#############

drg = haf.HQDeringmod(src)
sbl = get_y(src).std.Sobel().std.Binarize(10000).std.Maximum()
lms = get_y(src).std.Binarize(20000).std.Maximum()
drm = core.std.Expr([sbl, lms], "x y min").std.Maximum().std.Maximum().std.Maximum()
drg = src.std.MaskedMerge(drg, drm)
drg = drg.neo_f3kdb.Deband(range=20, y=64, cb=64, cr=64, grainy=0, grainc=0, sample_mode=4)
drg = adptvgrnMod(drg, luma_scaling=100, static=True, seed=3489)
src = awf.rfs(src, drg, "[199828 207003]")

src = src[:1010] + adptvgrnMod(src, size=1, luma_scaling=100, grainer=lambda s: pointgrn(s, size=1, strength=.2, cstrength=.1, seed=69))[1010:]

src = depth(src, 8)

#########################
# KOR theatrical scenes #
#########################

kor = depth(kor, 16)

# rounding error fix
kor = kor.std.Expr("x 128 +")

kor = kor.resize.Point(matrix_in_s="470bg", matrix_s="709", dither_type="error_diffusion")

kra = adptvgrnMod(kor, luma_scaling=25)
kra = depth(kra, 8)

krb = kor.neo_f3kdb.Deband(range=25, y=48, cb=48, cr=48, grainy=0, grainc=0, sample_mode=4)
msk = retinex_fdog(kor, 2.5).std.Binarize(10000).std.Maximum().std.Maximum().std.Inflate().std.Maximum()
krb = krb.std.MaskedMerge(kor, msk)
krb = adptvgrnMod(krb, size=0.8, sharp=100, luma_scaling=70, grainer=lambda c: pointgrn(c, 1/2.4, strength=.4, cstrength=.4))
krb = depth(krb, 8)

src = src[:55225] + kra[55225:55273] + src[55225:170037] + krb[170085:170161] + src[170037:]

src.set_output()
