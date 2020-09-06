import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from rekt import *
from adptvgrnMod import *
from bandmask import *
from vsutil import *
from autodbnd import autodbnd
core = vs.get_core()

#1 Import
src = core.ffms2.Source('/home/Videos/illusion/remux.mkv')

#2 Crop change bit depth and filter
src = src.std.Crop(left=220, right=220, top=0, bottom=0)
src = fvf.Depth(src, 16)
src = core.fb.FillBorders (src, left=0, right=1, top=1, bottom=1, mode="fillmargins")
src = rektlvls(src, rownum=[1,2,3,1077,1078], rowval=[20,-10,-2,-12,17], colnum=[0,1,2,1476,1477,1478], colval = [16,-10,-4,4,-3,-10])
src = awf.CropResize(clip=src, width = 988, left=0, right=1, bottom=1, top=1, aspect_ratio=4/3)

# Deband Fade-ins & Fade-outs
dbn = core.f3kdb.Deband(src, range=15, y=60, cb=0, cr=0, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=400)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Binarize(15000).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=1.0, strength=0.3, cstrength=0, luma_scaling=4, seed=3)
snf = fvf.rfs(src, grn, "[3643 3676] [3691 3745] [42870 42916] [48668 48709] [50519 50540] [64730 64758] [82462 82499] [104729 104846] [128166 128213] [128224 128262] [142430 142501] [146774 146842] [148002 148153] [152814 152867] [154049 154094] [154099 154130] [159724 159762] [159770 159798] [163243 163302]")

#Deband #250
dbn = core.f3kdb.Deband(src, range=20, y=56, cb=0, cr=0, grainy=0, grainc=0, output_depth=16)
bms = bandmask(src, thr=250)
mrg = core.std.MaskedMerge(src, dbn, bms)
msk = kgf.retinex_edgemask(src).std.Binarize(5800).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, src, msk)
grn = adptvgrnMod(mrg, size=1.2, strength=.5, sharp=30, cstrength=0, luma_scaling=100, seed=3)
snf = fvf.rfs(snf, grn, "[3761 5002] [33202 33226] [57492 57649] [80169 80204] [121783 122008] [125149 125838] [125892 126151] [126156 126222] [135470 135668] [135831 136065] [136095 137144] [137173 138289] [148300 148323] [152382 152417] [156052 156182] [158214 158374] [159158 159326]")
out = fvf.Depth(snf, 8)

#extract = awf.ExtractFramesReader(out, csvfile='/home/Videos/illusion/deband_scripts/frames.txt')
out.set_output()
