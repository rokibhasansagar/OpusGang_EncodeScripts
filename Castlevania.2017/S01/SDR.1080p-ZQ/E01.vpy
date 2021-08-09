from functools import partial

import vapoursynth as vs
core = vs.core

import vsutil
import lvsfunc
import vardefunc
import awsmfunc
import havsfunc

import Castlevania_common

from adptvgrnMod import adptvgrnMod
from vsTAAmbk import TAAmbk

GBR = core.lsmas.LWLibavSource('')
WEB = core.ffms2.Source('')

GBR = vsutil.depth(GBR, 16)[:35109]
WEB = vsutil.depth(WEB, 16)[:35109]

GBR = awsmfunc.fixlvls(GBR, preset=1)

OP, ED = (10141, 11708), (33670, GBR.num_frames - 1)

mask = lvsfunc.mask.detail_mask(GBR, rad=4, brz_a=0.025, brz_b=0.025)

hybrid = core.std.MaskedMerge(WEB, GBR, mask)
hybrid = core.std.ShufflePlanes([hybrid, GBR], [0,1,2], vs.YUV)
hybrid = lvsfunc.rfs(hybrid, GBR, ranges=[(25876, 26237), (26975, 27121), (27465, 27535), (27668, 27942), (28089, 30236)])

aa_1 = TAAmbk(hybrid, aatype='Eedi2', aatypeu='Eedi2', aatypev='Eedi2', mtype=3, opencl=False, stabilize=0, showmask=False)
aa_1 = lvsfunc.rfs(aa_1, hybrid, ranges=[OP])

aa_2 = Castlevania_common.sraa(hybrid, rfactor=[1.8]*3)
aa_2 = lvsfunc.rfs(aa_1, aa_2, ranges=[(17755, 18228), (19314, 19568)])

deblock = Castlevania_common.dpir_deblock(aa_2, strength=35, task='deblock') # magic not of this earth
deband = vardefunc.deband.dumb3kdb(deblock, threshold=30, radius=20, sample_mode=4, use_neo=True)
merge = core.std.MaskedMerge(deband, deblock, mask)
contra = Castlevania_common.csharp(merge, aa_1, mode=20)

grain = adptvgrnMod(contra, strength=.2, luma_scaling=6, static=True, seed=444)

opening_mask = core.std.Prewitt(vsutil.get_y(hybrid), planes=0, scale=.5).morpho.Dilate(size=2).morpho.Erode(size=2)

obliterate_grain_1 = Castlevania_common.dpir_deblock(contra, strength=40, task='denoise')
obliterate_grain_1 = lvsfunc.rfs(contra, obliterate_grain_1, ranges=[(11033, 11643)])

obliterate_grain_2 = core.knlm.KNLMeansCL(contra, h=20, s=1, a=1, d=1, channels='Y')
obliterate_grain_2 = core.std.MaskedMerge(obliterate_grain_2, contra, opening_mask)
obliterate_grain_2 = lvsfunc.rfs(obliterate_grain_1, obliterate_grain_2, ranges=[(11644, 11708)])

obliterate_chroma = Castlevania_common.quick_resample(contra, partial(Castlevania_common.bm3d, sigma=[0,10,10]))
obliterate_chroma = core.std.ShufflePlanes([obliterate_grain_2, obliterate_chroma], [0,1,2], vs.YUV)
obliterate_chroma = core.std.MaskedMerge(obliterate_chroma, obliterate_grain_2, opening_mask)
obliterate_chroma = Castlevania_common.csharp(obliterate_chroma, obliterate_grain_2, mode=20)
obliterate_chroma = lvsfunc.rfs(obliterate_grain_2, obliterate_chroma, ranges=[(10731, 10768)])

opening_aa = TAAmbk(obliterate_chroma, aatype='Eedi2', aatypeu='Eedi2', aatypev='Eedi2', mtype=0, opencl=False, showmask=False)
opening_aa = Castlevania_common.csharp(opening_aa, obliterate_chroma, mode=21)
opening_grain = adptvgrnMod(opening_aa, strength=10, grain_chroma=False, seed=444, luma_scaling=0, show_mask=False)

opening = lvsfunc.rfs(grain, opening_grain, ranges=[OP])
ending = lvsfunc.rfs(opening, GBR, ranges=[ED])

vsutil.depth(ending, 10).set_output()