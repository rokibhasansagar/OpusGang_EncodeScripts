import vapoursynth as vs
import vsutil
from rekt import rektlvls, rekt_fast
import awsmfunc
import havsfunc
import vsTAAmbk as taa
import lvsfunc
from math import sqrt

core = vs.core
core.max_cache_size = 1024

card1, card2 = (3456, 3549), (14136, 14243)

jpn = core.lsmas.LWLibavSource(r'Berserk/E01/00000_JPN.m2ts')
jpn = core.std.Crop(jpn, left=240, right=240)
jpn = vsutil.depth(jpn, 16)

dln = rektlvls(jpn, rownum=[0,1,2,3,4,6,1078,1077,1076,1074,1072,1079], rowval=[39,13,-5,5,2,2,7,-3,2,-1,-1,20], colnum=[5,4,1,0,1439,1438,1437,1436,1435,1434,1433], colval=[1,-1,-5,34,50,3,-2,3,1,2,2])
dln = awsmfunc.bbmod(dln, planes=[1,2], top=2, bottom=2, left=2, right=2)

# Couldn't get prot_val to behave so this garbage makes a return
cnr = rekt_fast(dln, lambda x: core.fxb.FillBorders(x, top=1), left=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, top=1), right=1438)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, left=1), top=1078)
cnr = rekt_fast(cnr, lambda x: core.fxb.FillBorders(x, right=1), top=1078)

card = havsfunc.DeHalo_alpha(cnr, darkstr=1.5, rx=2.7, ry=1.5, brightstr=0)
card = havsfunc.HQDeringmod(card, mrad=3, mthr=100, nrmode=1, thr=50, show=False)
card = taa.TAAmbk(card, 6, 0, 0)
card = lvsfunc.rfs(cnr, card, ranges=[card1])

b32 = vsutil.depth(card, 32)
lineart = core.placebo.Resample(b32, width=960, height=720, filter='ewa_robidouxsharp', param1=6./(13.+7.*sqrt(2.)), param2=7./(2.+12.*sqrt(2.)))
halos = lvsfunc.mask.halo_mask(card, rad=1, thmi=25).resize.Bicubic(width=960, height=720, format=vs.GRAYS, filter_param_a=0, filter_param_b=0)
background = core.placebo.Resample(b32, width=960, height=720, filter='haasnsoft', blur=1.025)
merge = core.std.MaskedMerge(background, lineart, halos)

card = core.placebo.Resample(merge, width=960, height=720, filter='gaussian', blur=1.3)
card = lvsfunc.rfs(merge, card, ranges=[card1, card2])

out = vsutil.depth(card, 8).set_output()