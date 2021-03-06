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

text_1, text_2, text_3, text_4, text_5, text_6 = (73, 280), (475, 584), (589, 667), (679, 1069), (1358, 1575), (1576, 1678)

jpn = core.lsmas.LWLibavSource(r'Berserk/E01/00000_JPN.m2ts')[1776:3455]
jpn = core.std.Crop(jpn, left=240, right=240)
jpn = vsutil.depth(jpn, 16)

dln = rektlvls(jpn, rownum=[0,1,2,3,4,6,1078,1077,1076,1074,1072,1079], rowval=[39,13,-5,5,2,2,7,-3,2,-1,-1,20],
    colnum=[5,4,1,0,1439,1438,1437,1436,1435,1434,1433], colval=[1,-1,-2,34,50,3,-2,3,1,2,2])
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

msk_5 = core.tcanny.TCanny(y, sigma=1, t_h=65.0, t_l=40.0, op=0)
brize = core.std.Binarize(y, 95 << 8).std.Invert()
brize = vsutil.iterate(brize, lambda x: core.std.Maximum(x), 4)
msk_5 = core.std.Expr([msk_5, brize], 'x y -')
msk_5 = vsutil.iterate(msk_5, lambda x: core.std.Maximum(x), 15)

msk_6 = core.tcanny.TCanny(y, sigma=1, t_h=65.0, t_l=40.0, op=0)
brize = core.std.Binarize(y, 108 << 8).std.Invert()
brize = vsutil.iterate(brize, lambda x: core.std.Maximum(x), 4)
msk_6 = core.std.Expr([msk_6, brize], 'x y -')
msk_6 = vsutil.iterate(msk_6, lambda x: core.std.Maximum(x), 13)

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

halos = havsfunc.DeHalo_alpha(cnr)

scene_4 = core.std.MaskedMerge(cnr, halos, msk_4)
scene_4 = lvsfunc.rfs(scene_3, scene_4, ranges=[text_4])

scene_5 = core.std.MaskedMerge(cnr, halos, msk_5)
scene_5 = lvsfunc.rfs(scene_4, scene_5, ranges=[text_5])

scene_6 = core.std.MaskedMerge(cnr, halos, msk_6)
scene_6 = lvsfunc.rfs(scene_5, scene_6, ranges=[text_6])

haha_dering_go_brrr = havsfunc.HQDeringmod(scene_6)
haha_dering_go_brrr = lvsfunc.rfs(haha_dering_go_brrr, scene_1, ranges=[text_1])

out = vsutil.depth(haha_dering_go_brrr, 16, dither_type='none').set_output()