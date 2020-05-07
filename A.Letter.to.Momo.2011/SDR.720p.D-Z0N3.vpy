import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import mvsfunc as mvf
import kagefunc as kgf
import vsTAAmbk as taa
import sgvsfunc as sgf
import havsfunc as haf
import blazefunc as bzf
import scoll
from fag3kdb import *
from debandmask import *
from rekt import *
from black_detect import *
from adptvgrnMod import *
from MCAA import *
from nnedi3_rpow2 import *
from insaneAA import *
from vsutil import *

src = core.ffms2.Source("")

s16 = fvf.Depth(src, 16)

s16 = core.resize.Spline36(s16, 1280, 720)

dny = core.knlm.KNLMeansCL(s16, a=3, h=0.35, d=2, device_type='gpu', device_id=0, channels='Y')
dnuv = core.knlm.KNLMeansCL(dny, a=2, h=0.5, d=1, device_type='gpu', device_id=0, channels='UV')
sf = adptvgrnMod(dnuv, strength=0.2, size=1.1, sharp=70, static=True, luma_scaling=8, grain_chroma=True)

sf = haf.FineDehalo(sf, rx=2, thmi=80, thma=128, thlimi=25, thlima=100, darkstr=0.3, brightstr=0.4)

mask = kgf.retinex_edgemask(s16).std.Binarize(65500).std.Maximum().std.Inflate()

aa = taa.TAAmbk(sf, aatype=3, mtype=0, opencl=False)
aa = core.std.MaskedMerge(sf, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[2429 2595] [22294 22457]")

aa = taa.TAAmbk(s16, aatype=6, mtype=0, repair=16, opencl=False)
aa = core.std.MaskedMerge(sf, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[53867 54145] [144578 144648]")

aa = taa.TAAmbk(s16, aatype=2, mtype=0, opencl=False)
aa = core.std.MaskedMerge(sf, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[5521 5615] [6329 6656] [9304 9686] [97279 97385]")

dbmask = kgf.retinex_edgemask(s16).std.Binarize(25000).std.Inflate()
deband = fvf.GradFun3(dnuv, thr=1, mask=0, radius=10, smode=2)
merge = core.std.MaskedMerge(deband, s16, dbmask).grain.Add(var=.1, constant=True)
merge = adptvgrnMod(merge, strength=0.3, size=1.1, sharp=70, static=True, luma_scaling=11, grain_chroma=True)
sf = fvf.rfs(sf, merge, "[46697 46774]")

dbmask = kgf.retinex_edgemask(s16).std.Expr("x 30000 > x 10 * x ?")
deblock = haf.Deblock_QED(dnuv, quant1=30, quant2=32)
merge = core.std.MaskedMerge(deblock, dnuv, dbmask)
merge = adptvgrnMod(merge, strength=1.4, size=1.1, sharp=50, static=True, luma_scaling=10, grain_chroma=True)
sf = fvf.rfs(sf, merge, "[138724 138909]")

final = fvf.Depth(sf, 8)

final = fvf.rfs(final, fvf.Depth(s16, 8), "[0 748] [172890 173107]")

cf = core.edgefixer.ContinuityFixer(src.std.Crop(top=8, bottom=8), left=4, top=0, right=3, bottom=0, radius=100)
cf = core.fb.FillBorders(cf, top=2, bottom=1, right=2, left=4).std.Merge(cf, [1, 0]).resize.Spline36(1280, 708, src_top=1, src_height=1062).std.AddBorders(top=6, bottom=6)
final=fvf.rfs(final, cf, "[485 674]")

blank = core.std.BlankClip(src, 1280, 720, color=[16, 128, 128])
final = fvf.rfs(final, blank, "[747 748] [868 911] [54146 54148] [104064 104086] [147753 147820]")

final.set_output()

# zones
# 0,758,crf=11.8/119565,124694,crf=11.0/115400,117358,crf=11.0/131567,138723,crf=10.8/138724,138909,crf=10.0/138910,147752,crf=11.1