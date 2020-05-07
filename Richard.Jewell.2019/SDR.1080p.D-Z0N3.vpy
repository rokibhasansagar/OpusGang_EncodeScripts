import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import awsmfunc as awf
import kagefunc as kgf
import havsfunc as haf
from rekt import rektlvls, rekt_fast
from vsutil import get_y, split
from adptvgrnMod import *
from bandmask import *

src = core.ffms2.Source("/home/aicha/Videos/Richard.Jewell.2019.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-HiFi.mkv")

src = fvf.Depth(src, 16)

src = src.std.Crop(top=138, bottom=138)

fxb = rektlvls(src, rownum=[0, 803], rowval=[8, 8], colnum=[0, 1919], colval=[8, 8])

# crop
crp = fxb.std.Crop(left=258, right=258)
crp = awf.bbmod(crp, left=1, right=1, u=True, v=True, blur=20, thresh=[5000, 1000])
crp = awf.AddBordersMod(crp, left=258, right=258, lsat=1, rsat=.6)
snf = fxb.remap.Rfs(crp, mappings="[19470 20224] [49876 50060] [91088 91141] [91195 91446] [91780 92025] [97808 97930] [97995 98866] [100921 102601] [102702 103839] [105063 105291] [105321 105413] [105468 105536] [105615 105703] [105780 105863] [105921 106060]")

# fades
dbn = fxb.f3kdb.Deband(range=10, y=32, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
msk = kgf.retinex_edgemask(fxb).std.Binarize(5000).std.Inflate().std.Inflate()
mrg = dbn.std.MaskedMerge(fxb, msk)
grn = adptvgrnMod(mrg, size=1.2, sharp=40, strength=1.5, cstrength=1, luma_scaling=100, static=False, fade_edges=True, show_mask=False)
snf = snf.remap.Rfs(grn, mappings="[678 699] [132837 132843] [132857 132862] [132914 132922] [132927 132936] [132974 132985] [132996 133007] [133029 133040] [133052 133063] [133630 133643] [133653 133664] [134143 134151] [134161 134176] [179107 179161]")

# blocking
dbn = fxb.f3kdb.Deband(range=20, y=32, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
bms = bandmask(fxb, thr=200, pix=3, dec=2, mid=2, exp=6)
mrg = fxb.std.MaskedMerge(dbn, bms).std.MaskedMerge(fxb, msk)
grn = adptvgrnMod(mrg, size=1.15, sharp=40, strength=1.5, cstrength=1, luma_scaling=10, static=False, fade_edges=True, show_mask=False)
snf = snf.remap.Rfs(grn, mappings="[4588 4736] [6511 6636] [26000 26096] [43551 43608] [43721 43791] [44247 44303] [44763 44840] [46070 46128] [95005 95074] [116434 116745] [155209 155421]")

# soft
dbn = fxb.f3kdb.Deband(range=10, y=16, cb=16, cr=16, grainy=0, grainc=0, output_depth=16)
mrg = fxb.std.MaskedMerge(dbn, bms).std.MaskedMerge(fxb, msk)
grn = adptvgrnMod(mrg, size=1.15, sharp=40, strength=1.5, cstrength=1, luma_scaling=10, static=False, fade_edges=True, show_mask=False)
snf = snf.remap.Rfs(grn, mappings="[15244 15366] [15630 15721] [91022 91087] [91142 91194] [95246 95317]")

# chroma is a bit messy
grn = adptvgrnMod(fxb, size=1.15, sharp=40, strength=.5, cstrength=.5, luma_scaling=10, static=False, fade_edges=True, show_mask=False)
snf = snf.remap.Rfs(grn, mappings="[48696 48787] [95204 95245]")

# variable crop
vcr = snf.fb.FillBorders(bottom=1, mode="fillmargins").cf.ContinuityFixer(bottom=[3, 3, 3], radius=5)
snf = fvf.rfs(snf, vcr, "43591 [43599 43600]")

ref = awf.bbmod(snf, bottom=4, blur=3, u=False, v=False)
flb = ref.fb.FillBorders(bottom=2, mode="fillmargins")
vcr = ref.fb.FillBorders(bottom=4, mode="fillmargins").std.Merge(flb, [1, 0])
vcr = snf.edgefixer.Reference(vcr, bottom=4, radius=6).std.Merge(vcr, [0, 1])
vcr = awf.bbmod(vcr, bottom=4, u=False, v=False)
vcr = rekt_fast(vcr, lambda c: c.fb.FillBorders(bottom=2, mode="fillmargins"), right=500)
snf = fvf.rfs(snf, vcr, "43601")

# ringing
drn = haf.HQDeringmod(fxb, drrep=0, mrad=2, msmooth=2, elast=3, sharp=0)
rgb = core.std.Expr(split(awf.DelFrameProp(fxb.resize.Bilinear(format=vs.RGBS, matrix_in_s="709"))), "x {0} - abs {1} > y {0} - abs {1} > and z {0} - abs {1} > and 1 0 ?".format(.3, .2)).std.Maximum()
drn = fxb.std.MaskedMerge(drn, rgb.resize.Point(format=vs.GRAY16))
snf = snf.remap.Rfs(drn, mappings="[149643 149803] [150321 150430] [150869 151133] [151210 151464] [151578 151914] [152008 152229] [152362 152643] [152743 153109] [153209 153387] [153516 153675] [153714 153949] [154045 154507] [154608 154795]")

grn = adptvgrnMod(snf, size=1.1, strength=.2, cstrength=0, sharp=40, static=False, fade_edges=True, luma_scaling=50)

out = fvf.Depth(snf, 8)

out.set_output()