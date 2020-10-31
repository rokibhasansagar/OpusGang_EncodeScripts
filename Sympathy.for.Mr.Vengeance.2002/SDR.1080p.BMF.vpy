import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
import vsutil as vsu
import havsfunc as haf
from rekt import *
from adptvgrnMod import *
from bandmask import *
from autodbnd import autodbnd
core = vs.get_core()

#1 Import / crop / bit depth
src = core.lsmas.LWLibavSource('Sympathy.for.Mr.Vengeance.2002.GBR.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-decibeL.mkv')
src = src.std.Crop(top=132, bottom=132)
dep = depth(src, 16)

#2 Debanding Intro
dbn = core.f3kdb.Deband(dep, range=20, y=60, cb=40, cr=40, grainy=0, grainc=0, output_depth=16)
msk = kgf.retinex_edgemask(dep).std.Binarize(55000).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, dep, msk)
grn = adptvgrnMod(mrg, size=1.1, strength=0.8, sharp=60, cstrength=.4, luma_scaling=100, seed=444)
snf = awf.rfs(dep, grn, "[0 539]")

#3 Debanding

dbn = core.f3kdb.Deband(dep, range=15, y=58, cb=40, cr=40, grainy=0, grainc=0, output_depth=16)
bms = bandmask(dep, thr=200)
mrg = core.std.MaskedMerge(dep, dbn, bms)
msk = kgf.retinex_edgemask(dep).std.Binarize(7500).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, dep, msk)
grn = adptvgrnMod(mrg, size=1.1, strength=0.8, sharp=60, cstrength=.4, luma_scaling=95, seed=3)
snf = awf.rfs(snf, grn, "[715 1156][2240 2360][1174 1276][1296 1376][3846 4556][4703 4843][14244 14317][14321 14359][24832 25046][29406 29457][37729 37765][37827 38815][67169 67434][67467 67605][68212 68246][108234 108709][108814 108948][118073 118117][139617 139659][141081 141127][142060 142117][147566 147721][148108 149909][156034 156201][167064 167131]")


#4 Static grain and debanding for bright scenes
dbn = core.f3kdb.Deband(dep, range=15, y=58, cb=40, cr=40, grainy=0, grainc=0, output_depth=16)
bms = bandmask(dep, thr=200)
mrg = core.std.MaskedMerge(dep, dbn, bms)
msk = kgf.retinex_edgemask(dep).std.Binarize(7500).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, dep, msk)
grn = sizedgrn(mrg, size=1.1, strength=0.8, cstrength=0.4, sharp=60, seed=4)
snf = awf.rfs(snf, grn, "[40060 40273][109289 109539]")

#5 Tested a combination of sized grain and adptvgrnMod on one scene
dbn = core.f3kdb.Deband(dep, range=15, y=56, cb=48, cr=48, grainy=0, grainc=0, output_depth=16)
dbn = sizedgrn(dbn, strength=1.1, cstrength=0.4, size=0.8, sharp=60, static=True, seed=37)
msk = kgf.retinex_edgemask(dep).std.Binarize(7500).std.Inflate()
bms = awf.bandmask(dep, thr=200)
dbn = dep.std.MaskedMerge(dbn, bms)
dbn = dbn.std.MaskedMerge(dep, msk)
dbn = adptvgrnMod(dbn, size=1.1, strength=0.8, cstrength=0.4, sharp=60, static=False, luma_scaling=70, seed=15)
snf = awf.rfs(snf, dbn, "[80634 84913]")

gbr = fvf.Depth(snf, 8)

#6 Taiwanese Source and Hybrid
twn = core.lsmas.LWLibavSource('Sympathy.for.Mr.Vengeance.2002.1080p.TWN.Blu-ray.AVC.DTS-HD.MA.5.1-nLiBRA').std.Crop(top=132, bottom=132)
twn = fvf.Depth(twn, 16)
dbn = core.f3kdb.Deband(twn, range=20, y=56, cb=30, cr=30, grainy=0, grainc=0, output_depth=16)
msk = kgf.retinex_edgemask(twn).std.Binarize(50000).std.Inflate().std.Inflate()
mrg = twn.std.MaskedMerge(twn, msk)
grn = haf.GrainFactory3(mrg, g1str=19.0, g2str=17.0, g3str=15.0, g1shrp=65, g2shrp=73, g3shrp=85, g1size=1.5, g2size=1.2, g3size=0.9, temp_avg=0, ontop_grain=0.0, th1=24, th2=56, th3=128, th4=160)
twn = fvf.Depth(grn, 8)
hyb = gbr[:65842] + twn[66350:66488] + gbr[65980:174088]

hyb.set_output()

