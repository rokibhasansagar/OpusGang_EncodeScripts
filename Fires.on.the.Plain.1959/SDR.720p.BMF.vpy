import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
import vsutil as vsu
from rekt import *
from adptvgrnMod import *
from bandmask import *
from autodbnd import autodbnd
core = vs.get_core()

#1 Import / crop / bit depth
src = core.ffms2.Source('/home/Videos/plain/remux.mkv')
src = src.std.Crop(top=138, bottom=138)
src = fvf.Depth(src, 16)

#2 General dirty lines
lvl = rektlvls(src, colnum=[1,2,1917,1918], colval=[19,-3,-1,20])
bb = awf.bbmod(lvl, left=2, right=2, thresh=None, blur=999, y=True, u=None, v=None)

#3 Crazy borders moving up and down the edge
lvl2 = rektlvls(src, colnum=[1917,1918], colval=[-1,20])
bb2 = awf.bbmod(lvl2, right=2, thresh=None, blur=999, y=True, u=None, v=None)
fix = core.edgefixer.ContinuityFixer(bb2, left=[5, 0, 0], top=0, right=0, bottom=0, radius=5)
y = vsu.get_y(src)
msk = y.std.Binarize(60 << 8)
fix = src.std.MaskedMerge(fix, msk)
snf = awf.rfs(bb, fix, "[514 644] [62982 63582] [63652 63987] [64123 64684] [64782 65285] [96014 96085] [105242 105470] [105529 105678] [106325 107628] [108399 109458] [112654 112885] [113026 113448]")

#4 Resize
res = awf.CropResize(clip=snf, width=1280, left=1, right=1)

#5 Debanding
dbn = core.f3kdb.Deband(res, range=15, y=48, cb=0, cr=0, grainy=0, grainc=0, output_depth=16)
bms = bandmask(res, thr=200)
mrg = core.std.MaskedMerge(res, dbn, bms)
msk = kgf.retinex_edgemask(res).std.Binarize(5000).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, res, msk)
grn = adptvgrnMod(mrg, size=0.9, strength=1.0, sharp=40, cstrength=0, luma_scaling=6, seed=3)
snf = awf.rfs(res, grn, "[12618 12665] [19615 19806] [19838 19929] [19973 20212] [20830 20958] [24494 24565] [56836 56879] [64739 64779] [67704 67744] [69147 69208] [69883 69952] [70506 70552] [71711 71750] [72006 72116] [73034 73101] [75430 75477] [76708 76750] [76814 76910] [79255 79319] [79423 79478] [79720 79766] [80288 80324] [96562 96598] [98151 98191] [100338 100382] [100500 100544] [106944 106991] [113485 113540] [114615 114654] [118805 118839] [125060 125109] [125654 125688] [137059 137121]")

out = fvf.Depth(snf, 8)
out.set_output()
