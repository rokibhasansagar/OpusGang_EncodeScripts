import vapoursynth as vs
import vsutil
import awsmfunc
import havsfunc
import lvsfunc
import G41Fun_PR
import vardefunc
import os
import adjust

import vsTAAmbk as taa

from math import sqrt
from rekt import rektlvls, rekt_fast
from muvsfunc import BlindDeHalo3
from mvsfunc import BM3D
from dysfunctional import FDOG
from functools import partial
from adptvgrnMod import *
from G41Fun_PR import DetailSharpen

core = vs.core
core.max_cache_size = 3072

card_1, card_2, card_3 = (3456, 3551), (19752, 19871), (33912, 33983)
op_start, op_end = 1776, 3456
damaged_start, damaged_end = 16513, 16800

jpn = core.ffms2.Source('00003.m2ts', threads=1)
jpn = core.std.Crop(jpn, left=240, right=240)
jpn = vsutil.depth(jpn, 16)

dln = rektlvls(jpn, rownum=[0,1,2,3,4,6,1078,1077,1076,1074,1072,1079], rowval=[39,13,-5,5,2,2,7,-3,2,-1,-1,20], 
    colnum=[5,4,1,0,1439,1438,1437,1436,1435,1434,1433], colval=[1,-1,-5,34,50,3,-2,3,1,2,2])
dln = awsmfunc.bbmod(dln, planes=[1,2], top=2, bottom=2, left=5, right=2)
dln = lvsfunc.rfs(dln, core.cf.ContinuityFixer(dln, left=[8,8,8], right=[8,8,8], radius=200), ranges=[(16800, 16801)])

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
hlos = havsfunc.FineDehalo(plane[0], darkstr=0, rx=2)
hlos = lvsfunc.rfs(hlos, plane[0], ranges=[(op_start, op_end)])

Y = core.placebo.Resample(vsutil.depth(hlos, 32), width=960, height=720, filter='ewa_robidouxsharp', param1=6./(13.+7.*sqrt(2.)), param2=7./(2.+12.*sqrt(2.)), linearize=False, sigmoidize=False)
Y = vsutil.depth(Y, 16, dither_type='none')

pre = core.bilateral.Bilateral(Y, sigmaS=2)
msk_1 = core.adg.Mask(core.std.PlaneStats(Y), luma_scaling=24).std.Invert()
msk_2 = FDOG(pre, sigma=1, div=2)
msk_2 = vsutil.iterate(msk_2, lambda x: core.std.Inflate(x), 2)

dnr = havsfunc.SMDegrain(Y, plane=0, RefineMotion=True, thSAD=115)
adptv = core.std.MaskedMerge(Y, dnr, msk_1)
edges = core.std.MaskedMerge(adptv, Y, msk_2)

Cb = core.placebo.Resample(plane[1], 480, 360, filter='ewa_robidoux', linearize=False, sigmoidize=False, sx=-0.125)
Cr = core.placebo.Resample(plane[2], 480, 360, filter='ewa_robidoux', linearize=False, sigmoidize=False, sx=-0.125)

merge = vsutil.join([edges, Cb, Cr])

#dvd = core.d2v.Source('/DVD/BERSERK_DISC_5.d2v')
#dvd = core.vivtc.VFM(dvd, order=1).vivtc.VDecimate()[25855:26142]

dir = r"/DVD/rotated/"
rotated = [dir + rotated for rotated in os.listdir(dir)]
dvd = core.imwri.Read(rotated, firstnum=0)
dvd = core.std.AssumeFPS(dvd, fpsnum=24000, fpsden=1001)

dvd_crp = core.std.Crop(dvd, top=10, bottom=18, left=12, right=22)
dvd_rsz = core.resize.Bicubic(dvd_crp, format=vs.YUV420P16, matrix_s='470bg')

y, u, v = vsutil.split(dvd_rsz)
y = vardefunc.fsrcnnx_upscale(y, width=960, height=720, lmode=1, shader_file='FSRCNNX_x2_56-16-4-1.glsl')
u = core.resize.Bicubic(u, width=480, height=360, filter_param_a=1/3, filter_param_b=1/3)
v = core.resize.Bicubic(v, width=480, height=360, filter_param_a=1/3, filter_param_b=1/3)
yuv = vsutil.join([y, u, v])

dvd_lvl = awsmfunc.fixlvls(yuv, gamma=1, min_in=[16, 16], min_out=[2, 18], max_in=[255, 255], max_out=[220, 240])
dvd_lvl = adjust.Tweak(dvd_lvl, sat=0.75)
dvd_clr = vsutil.iterate(dvd_lvl, partial(core.resize.Point, matrix_in_s='709', matrix_s='470bg'), 2)

dvd_cln = core.bilateral.Bilateral(dvd_clr, planes=[0,1,2])
dvd_shp = DetailSharpen(dvd_cln, z=2)
dvd_lol = core.warp.AWarpSharp2(dvd_shp, blur=1, chroma=0, depth=8, planes=[1,2])
dvd_lol = core.std.MaskedMerge(dvd_lol, taa.TAAmbk(dvd_lol, aatype='Nnedi3SangNom'), core.std.Prewitt(dvd_lol, [0,1,2], 2.5))
dvd_hlo = havsfunc.FineDehalo(dvd_lol, darkstr=0)
dvd_grn = adptvgrnMod(dvd_hlo, strength=1.0, size=1.3, sharp=65, luma_scaling=0, seed=0)

dvd_mrg = merge[0:damaged_start] + dvd_grn + dvd_grn[283:284]*2 + merge[damaged_end:]

card = core.placebo.Resample(dvd_mrg, width=960, height=720, filter='gaussian', blur=1.3)
card = lvsfunc.rfs(dvd_mrg, card, ranges=[card_2, card_3])

out = vsutil.depth(card, 8).set_output()