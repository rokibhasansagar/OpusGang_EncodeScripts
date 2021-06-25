import vapoursynth as vs
core = vs.core

import vsutil
import vardefunc
import lvsfunc
import dysfunctional

from nnedi3_rpow2 import * 
from havsfunc import FineDehalo, SMDegrain
from adptvgrnMod import *
from rekt import rekt_fast

from typing import Optional

OPS, OPE, EDS = 3453, 5610, 31553

src = core.lsmas.LWLibavSource('')
src = vsutil.depth(src, 32)

descale = lvsfunc.scale.descale(src, width=1500, height=844, kernel=lvsfunc.kernels.Bilinear(), upscaler=None)

emask = vardefunc.mask.FreyChenG41().get_mask(descale).std.Inflate()

denoise = dysfunctional.bm3dGPU(descale, ref=lambda x: lvsfunc.util.quick_resample(x, function=lambda m: SMDegrain(m, tr=2, thSAD=115, plane=0)), sigma=2, profile='lc')
denoise = core.std.MaskedMerge(denoise, descale, emask)

deband = lvsfunc.util.quick_resample(denoise, function=lambda x: vardefunc.deband.dumb3kdb(x, threshold=[32,0,0]))
deband = core.std.MaskedMerge(deband, denoise, emask)
    
aa = core.std.MaskedMerge(rekt_fast(deband, fun=lambda x: lvsfunc.aa.upscaled_sraa(x), top=deband.height - 600, left=deband.width - 500), deband, emask)
aa = lvsfunc.rfs(deband, aa, ranges=(13085, 13131))

upscale = nnedi3_rpow2(aa, rfactor=2, width=1920, height=1080, kernel="spline64", upsizer='nnedi3cl')
upscale = core.std.ShufflePlanes([upscale, src], [0,1,2], vs.YUV)

chroma = lvsfunc.util.quick_resample(upscale, function=lambda x: vardefunc.deband.dumb3kdb(x, radius=20, threshold=[0,32,32]))
grain = dysfunctional.coolgrain(chroma, strength=[2,0], radius=2, luma_scaling=16, cutoff=16.5, divby=1.3, filter='robidoux')

op_halos = lvsfunc.rfs(grain, FineDehalo(src, darkstr=0, brightstr=.6, rx=2, ry=3, edgeproc=2, showmask=0), ranges=[(OPS, OPE)])
merge = core.std.MaskedMerge(grain, op_halos, lvsfunc.scale.descale_detail_mask(src, core.placebo.Resample(descale, width=src.width, height=src.height, filter='triangle'), threshold=0.05).std.Maximum())
vsutil.depth(merge, 10).set_output()