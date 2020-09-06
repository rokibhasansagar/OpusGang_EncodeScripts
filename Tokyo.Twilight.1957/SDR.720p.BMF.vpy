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
src = core.ffms2.Source('/home/Videos/tokyo/remux.mkv')
src = src.std.Crop(left=220, right=220)
src = fvf.Depth(src, 16)

##2 Dirty lines
src = rektlvls(src, rownum=[0], rowval=[2], colnum=[0, 1478], colval=[4, 6])

#3 Resize
src = awf.CropResize(clip=src, width=986, right=1)

#4 Debanding
dbn = core.f3kdb.Deband(src, range=15, y=55, cb=0, cr=0, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=200)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Binarize(6500).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=.6, strength=1.7, sharp=50, cstrength=0, luma_scaling=75, seed=3)
snf = awf.rfs(src, grn, "[54755 55029] [97289 97447] [154229 154355] [154452 154509] [154573 154829] [154856 154910] [154916 155221] [155269 155321] [155361 155483] [155547 155858]")

out2 = fvf.Depth(snf, 8)
out2.set_output()
