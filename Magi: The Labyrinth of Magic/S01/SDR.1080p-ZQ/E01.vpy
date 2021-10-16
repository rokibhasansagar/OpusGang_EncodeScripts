import vapoursynth as vs
import lvsfunc
import vsutil

from vsffdnet import FFDNet
from magi_common import frequency_merge
from dysfunctional import CoolDegrain
from vardefunc.scale import fsrcnnx_upscale
from lvsfunc.scale import descale_detail_mask as descale_mask
from adptvgrnMod import adptvgrnMod

core = vs.core

src = core.lsmas.LWLibavSource('[BDMV][アニメ][130130] マギ Vol.01/DISC1/BDROM/BDMV/STREAM/00000.m2ts')[0:35196]

src32 = vsutil.depth(src, 32)
src32_y = vsutil.get_y(src32)

bicubic = 1/3, 1/3
descale = lvsfunc.scale.descale(src32_y, height=720,
                                kernel=lvsfunc.kernels.Bicubic(b=bicubic[0], c=bicubic[1]),
                                upscaler=None)
# Fake RGB
denoise_y_pre = core.std.ShufflePlanes(descale, [0], vs.RGB)
denoise_y_rgb = FFDNet(denoise_y_pre, strength=3, device_type='cuda')
denoise_y = core.std.ShufflePlanes(denoise_y_rgb, [0], vs.GRAY)

merge_y = frequency_merge(lo=CoolDegrain(descale, thSAD=48, planes=[0]),
                            hi=denoise_y, thresh=10)
merge_c = frequency_merge(lo=src32, hi=core.knlm.KNLMeansCL(src32, h=1.0, d=2, a=1, s=4,
                                                            channels='UV'), thresh=1)

mask_ref = core.resize.Bicubic(descale, width=src32_y.width, height=src32_y.height,
                               filter_param_a=bicubic[0], filter_param_b=bicubic[1])
mask = descale_mask(src32_y, mask_ref, threshold=0.05).std.Limiter(min=0, max=1)

upscale = fsrcnnx_upscale(merge_y, width=1920, height=1080,
                          profile='slow', strength=88,
                          shader_file='FSRCNNX_x2_56-16-4-1.glsl')
upscale = core.std.ShufflePlanes([upscale, merge_c], [0, 1, 2], vs.YUV)
upscale = core.std.MaskedMerge(upscale, src32, mask)

grain = adptvgrnMod(upscale, strength=0.2, static=True)
out = core.fmtc.bitdepth(grain, bits=10, dmode=8).set_output()