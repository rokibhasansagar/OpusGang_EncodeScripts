import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
from adptvgrnMod import *
from rekt import *
from bandmask import *

src = core.ffms2.Source("/home/aicha/Videos/mv/Blood, Sweat & Tears 피땀눈물_BTS 방탄소년단-187617728.mov")
hbd = fvf.Depth(src, 16)
crp = hbd.std.Crop(top=134, bottom=134)

acr = crp.std.Crop(bottom=4)
ycr = acr.std.ShufflePlanes(0, vs.GRAY)
ucr = acr.std.ShufflePlanes(1, vs.GRAY)
vcr = acr.std.ShufflePlanes(2, vs.GRAY)

ufx = rektlvls(ucr, [ucr.height - 2], [-6.5], prot_val=0).fb.FillBorders(top=1, mode="fillmargins").cf.ContinuityFixer(bottom=1, radius=3)
vfx = vcr.cf.ContinuityFixer(bottom=3, radius=5).fb.FillBorders(bottom=2, top=1, mode="fillmargins")
lvl = rektlvls(ycr, [ycr.height - 2, ycr.height - 3], [-30, -6], prot_val=10)
fmg = lvl.cf.ContinuityFixer(bottom=2, radius=3).fb.FillBorders(bottom=1, top=1, mode="fillmargins")

acr = core.std.ShufflePlanes([fmg, ufx, vfx], [0, 0, 0], vs.YUV)
rsz = acr.resize.Spline36(format=vs.YUV420P16)
adb = rsz.std.AddBorders(bottom=4)

fmg = crp.fb.FillBorders(top=1, bottom=1, mode="fillmargins")
rsz = fmg.resize.Spline36(format=vs.YUV420P16)

trm = rsz.std.Trim(0, 5232) + adb.std.Trim(5233, 5260) + rsz.std.Trim(5261)

dbn = trm.f3kdb.Deband(y=64, cr=0, cb=0, range=6, grainy=0, grainc=0, output_depth=16)
msk = kgf.retinex_edgemask(trm).std.Binarize(11000).std.Maximum().std.Inflate()
mrg = dbn.std.MaskedMerge(trm, msk)
grn = adptvgrnMod(mrg, size=1.4, sharp=90, luma_scaling=4, grainer=lambda x: core.grain.Add(x, var=1.6, uvar=1.0, constant=False))
snf = fvf.rfs(trm, grn, "[1612 1616] [7779 7794]")

bmk = bandmask(trm, 200).std.Crop(top=500).std.AddBorders(top=500)
grn = adptvgrnMod(dbn, size=1.3, sharp=90, luma_scaling=4, grainer=lambda x: core.grain.Add(x, var=.4, uvar=.3, constant=False))
mrg = trm.std.MaskedMerge(grn, bmk)
snf = fvf.rfs(snf, mrg, "[7964 7980] [8009 8018]")

bmk = bandmask(trm, thr=300)
lmk = trm.std.ShufflePlanes(0, vs.GRAY).std.Binarize(55000).std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum()
gmk = core.std.Expr([bmk, lmk], "x y -").std.Crop(bottom=225, top=100).std.AddBorders(bottom=225, top=100).std.Maximum().std.Inflate().std.Maximum().std.Inflate().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate().std.Inflate()
gmk = kgf.iterate(gmk, core.std.Maximum, 3)
grn = adptvgrnMod(trm, size=1.2, sharp=80, luma_scaling=1, grainer=lambda x: core.grain.Add(x, var=1.4, uvar=1.0, constant=False))
mrg = core.std.MaskedMerge(trm, grn, gmk)
snf = fvf.rfs(snf, mrg, "[7579 7778]")

fdh = hvf.FineDehalo(trm)
grn = adptvgrnMod(fdh, size=1.2, sharp=80, luma_scaling=8, grainer=lambda x: core.grain.Add(x, var=.8, uvar=.5, constant=False))
snf = fvf.rfs(snf, grn, "[6523 6591]") # zone this

out = fvf.Depth(snf, 10)

# statue - don't think it's worth doing
# ? [8182 8197]

# red 5595,5984

out.set_output()