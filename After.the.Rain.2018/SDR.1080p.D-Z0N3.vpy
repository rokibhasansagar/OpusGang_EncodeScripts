import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
from vsutil import get_y
from adptvgrnMod import *
from rektlvls import *

src = core.ffms2.Source("")

src = core.std.Crop(src, top=136, bottom=136)

src = fvf.Depth(src, 16)

src = rektlvls(src, [0, 807], [45, 40])

src = core.edgefixer.ContinuityFixer(src, left=0, top=[0, 2, 2], right=0, bottom=0, radius=40)

b16 = src

y = get_y(b16)
w = y.std.Binarize(55000)

wmax = w.std.Maximum().std.Maximum().std.Maximum().std.Inflate()
wmax1 = w.std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Inflate()
wmin = w.std.Deflate()
wmask = core.std.Expr([wmax, wmin], "x y -").std.Inflate().std.Inflate()
wmask1 = core.std.Expr([wmax1, wmin], "x y -").std.Inflate().std.Inflate()

dr = hvf.HQDeringmod(b16, ringmask=core.std.BlankClip(y, color=[65535]), drrep=0)
dh = hvf.DeHalo_alpha(dr, darkstr=0.8, lowsens=40, highsens=80)
drh = core.std.MaskedMerge(b16, dh, wmask)

drh1 = core.std.MaskedMerge(b16, dh, wmask1)
drhb = drh1.deblock.Deblock(30)
drhbr = hvf.HQDeringmod(drhb, drrep=23)

drg = adptvgrnMod(drhbr, strength=2, size=1, static=False, luma_scaling=1)

sf = fvf.rfs(src, drh, "[2360 3217] [3505 3575]")
sf = fvf.rfs(sf, drg, "[2199 2359]")

db = b16.f3kdb.Deband(range=15, y=64, cb=64, cr=64, grainy=24, grainc=24, keep_tv_range=True, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Maximum().std.Maximum().std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
sf = fvf.rfs(sf, merged, mappings="[25 262]")

db = b16.f3kdb.Deband(range=20, y=64, cb=64, cr=64, grainy=48, grainc=48, dynamic_grain=True, keep_tv_range=True, output_depth=16)
mask = kgf.retinex_edgemask(y).std.Binarize(10000).std.Inflate().std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
sf = fvf.rfs(sf, merged, "[18787 18885] [19187 19310]")

mask = kgf.retinex_edgemask(y).std.Maximum().std.Maximum().std.Inflate().std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
sf = fvf.rfs(sf, merged, "[58947 59011] [59082 60349] [76011 76305] [94907 95025]")

db = b16.f3kdb.Deband(range=20, y=32, cb=32, cr=32, grainy=48, grainc=48, dynamic_grain=True, keep_tv_range=True, output_depth=16)
mask = kgf.retinex_edgemask(y).std.Expr("x 5000 > x 2 * x ?").std.Maximum().std.Maximum().std.Inflate().std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
merged = adptvgrnMod(merged, size=0.8, strength=1, static=False, luma_scaling=0)
sf = fvf.rfs(sf, merged, "[103942 104261]")

grn = adptvgrnMod(sf, strength=0.4, size=0.8, static=False, luma_scaling=8)

cre = hvf.HQDeringmod(sf)
cre = hvf.FineDehalo(cre, rx=1.5, thmi=120, thma=150, thlimi=50, thlima=100, showmask=0, excl=False)

sf = fvf.rfs(grn, cre, "[154436 160799]")

c = core.std.Crop(sf, left=228, top=172, right=910, bottom=198)
c = core.edgefixer.ContinuityFixer(c, left=4, top=4, bottom=3, right=5, radius=50)
adb = core.std.AddBorders(c, left=228, top=172, right=4, bottom=198)
cr = core.std.Crop(sf, left=1014)
sh = core.std.StackHorizontal([adb, cr])
sf = fvf.rfs(sf, sh, "[155461 156692]")

final = fvf.Depth(sf, 8)

final.set_output()