import vapoursynth as vs
core = vs.core

import vsutil
import vardefunc
import lvsfunc
import awsmfunc
import dysfunctional

from nnedi3_rpow2 import * 
from havsfunc import FineDehalo, SMDegrain
from rekt import rekt_fast

def rescale(clip: vs.VideoNode, width: int = 1280, height: int = 720, 
            kernel=lvsfunc.kernels.Bilinear(), threshold=0.05, show_mask=False) -> vs.VideoNode:
    
    descale = lvsfunc.scale.descale(clip, width=width, height=height, kernel=kernel, upscaler=None, mask=None)
    
    upscale = nnedi3_rpow2(descale, rfactor=2, width=1920, height=1080, kernel="spline64", upsizer='nnedi3cl')
    upscale = core.std.ShufflePlanes([upscale, clip], [0,1,2], vs.YUV)
    
    mask = lvsfunc.scale.descale_detail_mask(upscale, core.placebo.Resample(descale, width=clip.width, height=clip.height, 
                                                                            filter='triangle'), threshold=threshold)
    
    if show_mask is True: return mask
    else: return core.std.MaskedMerge(upscale, clip, mask)
    
    
def SM3D(clip: vs.VideoNode) -> vs.VideoNode:
    from vsutil import get_y
    from havsfunc import SMDegrain
    from lvsfunc.util import quick_resample
    
    luma = get_y(clip)
    ref = quick_resample(luma, lambda d: SMDegrain(d, tr=1, thSAD=115))
    
    denoise = core.bm3dcuda_rtc.BM3D(luma, ref, sigma=[3, 0, 0], fast=False, extractor_exp=8, transform_1d_s='DCT', 
                                     transform_2d_s='DCT', block_step=5, bm_range=9, radius=2, ps_num=2, ps_range=5)
    denoise = denoise.bm3d.VAggregate(radius=2, sample=1)
    return core.std.ShufflePlanes([denoise, clip], [0,1,2], vs.YUV)

# descale mask works just fine for ED
OPS, OPE = 3453, 5610

src = core.lsmas.LWLibavSource('')
src = vsutil.depth(src, 32)

descale = rescale(src, width=1500, height=844, kernel=lvsfunc.kernels.Bilinear())

emask = vardefunc.mask.FreyChenG41().get_mask(descale, multi=1).std.Inflate()

denoise = SM3D(descale)
denoise = core.std.MaskedMerge(denoise, descale, emask)

deband = lvsfunc.util.quick_resample(denoise, function=lambda x: vardefunc.deband.dumb3kdb(x, threshold=[32,24,24]))
deband = core.std.MaskedMerge(deband, denoise, emask)

aa = rekt_fast(deband, fun=lambda x: lvsfunc.aa.upscaled_sraa(x, rfactor=1.3), top=deband.height - 850, left=deband.width - 620)
aa = lvsfunc.rfs(deband, aa, ranges=(13085, 13131))

grain = dysfunctional.coolgrain(aa, strength=[2,0], radius=2, luma_scaling=16, cutoff=16.5, divby=1.3, filter='robidoux')

opening = core.std.MaskedMerge(grain, src, rescale(src, width=1500, height=844,
                                                   kernel=lvsfunc.kernels.Bilinear(), show_mask=True, threshold=0.03))
opening = lvsfunc.rfs(grain, FineDehalo(opening, darkstr=0, brightstr=.6, rx=2, ry=3, edgeproc=2, showmask=0), ranges=[(OPS, OPE)])
vsutil.depth(opening, 10).set_output()