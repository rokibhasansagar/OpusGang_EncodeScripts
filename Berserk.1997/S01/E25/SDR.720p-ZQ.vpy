from mvsfunc import GetPlane
import vapoursynth as vs
import vsutil
import awsmfunc
import havsfunc
import vsTAAmbk as taa
import lvsfunc
import vardefunc
import adjust

from math import sqrt
from rekt import rektlvls, rekt_fast
from dysfunctional import FDOG
from adptvgrnMod import *
from functools import partial
from G41Fun_PR import DetailSharpen

core = vs.core
core.max_cache_size = 2048

card_1, card_2 = (3216, 3311), (17424, 17543)
op_start, op_end = 1536, 3216
damaged_start, damaged_end = 19233, 19467

jpn = core.ffms2.Source('00004.m2ts', threads=1)
jpn = core.std.Crop(jpn, left=240, right=240)
jpn = vsutil.depth(jpn, 16)

dln = rektlvls(jpn, rownum=[0,1,2,3,4,6,1078,1077,1076,1074,1072,1079], rowval=[39,13,-5,5,2,2,7,-3,2,-1,-1,20], 
    colnum=[5,4,1,0,1439,1438,1437,1436,1435,1434,1433], colval=[1,-1,-5,34,50,3,-2,3,1,2,2])
dln = awsmfunc.bbmod(dln, planes=[1,2], top=2, bottom=2, left=2, right=2)

cnr = rekt_fast(dln, lambda x: core.fxb.FillBorders(x, top=1), left=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, top=1), right=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, left=1), top=1078)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, right=1), top=1078)

splice = cnr[0:op_start] + core.ffms2.Source('OP1.mkv', threads=1) + cnr[op_end:]

card = havsfunc.DeHalo_alpha(splice, darkstr=1.5, rx=2.7, ry=1.5, brightstr=0)
card = havsfunc.HQDeringmod(card, mrad=3, mthr=100, nrmode=15, thr=50, show=False)
card = taa.TAAmbk(card, 6, 4, 4)
card = lvsfunc.rfs(splice, card, ranges=[card_1])

plane = vsutil.split(card)

Y = core.placebo.Resample(vsutil.depth(plane[0], 32), width=960, height=720, filter='ewa_robidouxsharp', param1=6./(13.+7.*sqrt(2.)), param2=7./(2.+12.*sqrt(2.)), linearize=False, sigmoidize=False)
Y = vsutil.depth(Y, 16, dither_type='none')

pre = core.bilateral.Bilateral(Y, sigmaS=2)
msk_1 = core.adg.Mask(core.std.PlaneStats(Y), luma_scaling=36).std.Invert()
msk_2 = FDOG(pre, sigma=2, div=2)
msk_2 = vsutil.iterate(msk_2, lambda x: core.std.Inflate(x), 2)

dnr = havsfunc.SMDegrain(Y, plane=0, RefineMotion=True, thSAD=115)
adptv = core.std.MaskedMerge(Y, dnr, msk_1)
edges = core.std.MaskedMerge(adptv, Y, msk_2)

Cb = core.placebo.Resample(plane[1], 480, 360, filter='ewa_robidoux', linearize=False, sigmoidize=False, sx=-0.125)
Cr = core.placebo.Resample(plane[2], 480, 360, filter='ewa_lanczos', linearize=False, sigmoidize=False, sx=-0.125)

merge = vsutil.join([edges, Cb, Cr])

dvd = core.d2v.Source('BERSERK_DISC_6.d2v')
dvd = core.vivtc.VFM(dvd, order=1).vivtc.VDecimate()[130620:130849]
dvd = vsutil.depth(dvd, 16)

y, u, v = vsutil.split(dvd)
y = vardefunc.fsrcnnx_upscale(y, width=960, height=720, lmode=1, shader_file='FSRCNNX_x2_56-16-4-1.glsl')
u = core.resize.Bicubic(u, width=480, height=360, filter_param_a=1/3, filter_param_b=1/3)
v = core.resize.Bicubic(v, width=480, height=360, filter_param_a=1/3, filter_param_b=1/3)
yuv = vsutil.join([y, u, v])

dvd_lvl = awsmfunc.fixlvls(yuv, gamma=1, min_in=[16, 16], min_out=[2, 18])
dvd_lvl = adjust.Tweak(dvd_lvl, sat=0.75)

dvd_mrg = merge[0:damaged_start] + dvd_lvl + core.std.BlankClip(dvd_lvl)[:1]*5 + merge[damaged_end:]

card = core.placebo.Resample(dvd_mrg, width=960, height=720, filter='gaussian', blur=1.3)
card = lvsfunc.rfs(dvd_mrg, card, ranges=[card_2])

out = vsutil.depth(card, 8).set_output()
