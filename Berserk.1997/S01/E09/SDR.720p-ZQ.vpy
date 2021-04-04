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
core.max_cache_size = 3072

card_1, card_2, card_3 = (3096, 3191), (15552, 15671), (33912, 33983)
op_start, op_end = 1416, 3096

jpn = core.ffms2.Source('00003.m2ts', threads=1)
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

merge = vsutil.join([edges, Cb, Cr])

card = core.placebo.Resample(merge, width=960, height=720, filter='gaussian', blur=1.3)
card = lvsfunc.rfs(merge, card, ranges=[card_2, card_3])

out = vsutil.depth(card, 8).set_output()
#jpn.set_output(0)
#splice.set_output(1)