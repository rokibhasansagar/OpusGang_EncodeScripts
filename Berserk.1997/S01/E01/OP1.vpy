import vapoursynth as vs
import vsutil
import awsmfunc
import havsfunc
import vsTAAmbk as taa
import lvsfunc

from math import sqrt
from rekt import rektlvls, rekt_fast

core = vs.core
core.max_cache_size = 1024

text_1, text_2, text_3, text_4, text_5 = (73, 280), (475, 584), (589, 667), (679, 1069), (1358, 1678)

jpn = core.lsmas.LWLibavSource(r'00000_JPN.m2ts')[1776:3455]
jpn = core.std.Crop(jpn, left=240, right=240)
jpn = vsutil.depth(jpn, 16)

dln = rektlvls(jpn, rownum=[0,1,2,3,4,6,1078,1077,1076,1074,1072,1079], rowval=[39,13,-5,5,2,2,7,-3,2,-1,-1,20], colnum=[5,4,1,0,1439,1438,1437,1436,1435,1434,1433], colval=[1,-1,-5,34,50,3,-2,3,1,2,2])
dln = awsmfunc.bbmod(dln, planes=[1,2], top=2, bottom=2, left=2, right=2)

cnr = rekt_fast(dln, lambda x: core.fxb.FillBorders(x, top=1), left=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, top=1), right=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, left=1), top=1078)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, right=1), top=1078)

y = vsutil.get_y(cnr)
msk_1 = core.tcanny.TCanny(y, sigma=1, t_h=115.0, t_l=50.0, op=1)
msk_1 = vsutil.iterate(msk_1, lambda x: core.std.Maximum(x), 7)

msk_2 = core.tcanny.TCanny(y, sigma=0, t_h=115.0, t_l=50.0, op=1)
msk_2 = vsutil.iterate(msk_2, lambda x: core.std.Maximum(x), 8)

msk_3 = lvsfunc.mask.halo_mask(jpn, rad=1, thmi=70, thma=25)
msk_3 = vsutil.iterate(msk_3, lambda x: core.std.Maximum(x), 5)

msk_4 = core.tcanny.TCanny(y, sigma=1, t_h=65.0, t_l=40.0, op=0)
brize = core.std.Binarize(y, 65 << 8).std.Invert()
brize = vsutil.iterate(brize, lambda x: core.std.Maximum(x), 3)
msk_4 = core.std.Expr([msk_4, brize], 'x y -')
msk_4 = vsutil.iterate(msk_4, lambda x: core.std.Maximum(x), 15)

scene_1 = havsfunc.DeHalo_alpha(cnr, darkstr=.7, rx=1, ry=1.5, brightstr=1)
scene_1 = havsfunc.HQDeringmod(scene_1, mrad=3, mthr=100, nrmode=1, thr=25, show=False)
scene_1 = taa.TAAmbk(scene_1, 1)
scene_1 = core.std.MaskedMerge(cnr, scene_1, msk_1)
scene_1 = lvsfunc.rfs(cnr, scene_1, ranges=[text_1])

scene_2 = havsfunc.DeHalo_alpha(cnr, darkstr=0)
scene_2 = core.std.MaskedMerge(cnr, scene_2, msk_2)
scene_2 = lvsfunc.rfs(scene_1, scene_2, ranges=[text_2])

scene_3 = havsfunc.DeHalo_alpha(cnr, darkstr=.7)
scene_3 = core.std.MaskedMerge(cnr, scene_3, msk_3)
scene_3 = lvsfunc.rfs(scene_2, scene_3, ranges=[text_3])

scene_4 = havsfunc.DeHalo_alpha(cnr)
scene_4 = core.std.MaskedMerge(cnr, scene_4, msk_4)
scene_4 = lvsfunc.rfs(scene_3, scene_4, ranges=[text_4, text_5])

haha_dering_go_brrr = havsfunc.HQDeringmod(scene_4)
haha_dering_go_brrr = lvsfunc.rfs(haha_dering_go_brrr, scene_1, ranges=[text_1])

b32 = vsutil.depth(haha_dering_go_brrr, 32)
lineart = core.placebo.Resample(b32, width=960, height=720, filter='ewa_robidouxsharp', param1=6./(13.+7.*sqrt(2.)), param2=7./(2.+12.*sqrt(2.)))
background = core.placebo.Resample(b32, width=960, height=720, filter='haasnsoft', blur=1.025)
halos = lvsfunc.mask.halo_mask(haha_dering_go_brrr, rad=1, thmi=25).resize.Bicubic(width=960, height=720, format=vs.GRAYS, filter_param_a=0, filter_param_b=0)
merge = core.std.MaskedMerge(background, lineart, halos)

out = vsutil.depth(merge, 16, dither_type='none').set_output()