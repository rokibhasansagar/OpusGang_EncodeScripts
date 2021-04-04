import vapoursynth as vs
import vsutil
import awsmfunc
import havsfunc
import vsTAAmbk as taa
import lvsfunc

from math import sqrt
from rekt import rektlvls, rekt_fast
from dysfunctional import FDOG

core = vs.core
core.max_cache_size = 2048

card_1, card_2, card_3 = (2976, 3071), (17208, 17327), (33912, 33983)
op_start, op_end = 1296, 2976
dbl = [(16330, 16883), (16949, 17076), (17367, 17439), (17546, 17878), (18197, 18350), (18409, 18580), (18719, 18820), (18982, 19058), (19114, 19199), (19213, 19277), (19331, 19500), (19601, 20225), (20350, 20487), (20768, 20838), (21066, 21367), (21403, 21589), (23119, 23216), (24935, 24957), (25662, 25769)]
#zones = 8068,8127,crf=14.3/23217,23630,crf=14.3/24958,25000,crf=14.3/25148,25281,crf=14.3/25371,25387,crf=14.3/25466,25485,crf=14.3/25662,25769,crf=14.3/25996,27408,crf=14.3/27598,27693,crf=14.3/27787,28246,crf=14.3/28918,28996,crf=14.3

jpn = core.ffms2.Source('00000.m2ts', threads=1)
jpn = core.std.Crop(jpn, left=240, right=240)
jpn = vsutil.depth(jpn, 16)

dln = rektlvls(jpn, rownum=[0,1,2,3,4,6,1078,1077,1076,1074,1072,1079], rowval=[39,13,-5,5,2,2,7,-3,2,-1,-1,20], 
    colnum=[5,4,1,0,1439,1438,1437,1436,1435,1434,1433], colval=[1,-1,-5,34,50,3,-2,3,1,2,2])
dln = awsmfunc.bbmod(dln, planes=[1,2], top=2, bottom=2, left=2, right=2)

cnr = rekt_fast(dln, lambda x: core.fxb.FillBorders(x, top=1), left=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, top=1), right=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, left=1), top=1078)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, right=1), top=1078)

splice = cnr[0:op_start] + core.ffms2.Source('OP1.mkv') + cnr[op_end:]

card = havsfunc.DeHalo_alpha(splice, darkstr=1.5, rx=2.7, ry=1.5, brightstr=0)
card = havsfunc.HQDeringmod(card, mrad=3, mthr=100, nrmode=15, thr=50, show=False)
card = taa.TAAmbk(card, 6, 4, 4)
card = lvsfunc.rfs(splice, card, ranges=[card_1])

plane = vsutil.split(card)
hlos = havsfunc.FineDehalo(plane[0], darkstr=0, rx=2)
hlos = lvsfunc.rfs(hlos, plane[0], ranges=[(op_start, op_end)])

Y = core.placebo.Resample(vsutil.depth(hlos, 32), width=960, height=720, filter='ewa_robidouxsharp', param1=6./(13.+7.*sqrt(2.)), param2=7./(2.+12.*sqrt(2.)), linearize=False, sigmoidize=False)
Y = vsutil.depth(Y, 16, dither_type='none')

pre = core.bilateral.Bilateral(Y, sigmaS=1.5)
msk_1 = core.adg.Mask(core.std.PlaneStats(Y), luma_scaling=24).std.Invert()
msk_2 = FDOG(pre, sigma=1, div=2)
msk_2 = vsutil.iterate(msk_2, lambda x: core.std.Inflate(x), 2)

dnr = havsfunc.SMDegrain(Y, plane=0, RefineMotion=True, thSAD=115)
adptv = core.std.MaskedMerge(Y, dnr, msk_1)
edges = core.std.MaskedMerge(adptv, Y, msk_2)

Cb = core.placebo.Resample(plane[1], 480, 360, filter='ewa_robidoux', linearize=False, sigmoidize=False, sx=-0.125)

Cr = core.placebo.Resample(plane[2], 480, 360, filter='ewa_robidoux', linearize=False, sigmoidize=False, sx=-0.125)
Cr_pre = core.knlm.KNLMeansCL(Cr, channels='Y')
Cr_dbl = core.f3kdb.Deband(Cr_pre, seed=444, dynamic_grain=False)
Cr_dbl = core.std.MaskedMerge(Cr_dbl, Cr, core.resize.Bicubic(msk_2, 480, 360, filter_param_a=0, filter_param_b=0))
Cr = lvsfunc.rfs(Cr, Cr_dbl, ranges=dbl)

merge = vsutil.join([edges, Cb, Cr])

card = core.placebo.Resample(merge, width=960, height=720, filter='gaussian', blur=1.3)
card = lvsfunc.rfs(merge, card, ranges=[card_2, card_3])

out = vsutil.depth(card, 8).set_output()