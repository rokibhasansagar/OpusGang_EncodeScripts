import vapoursynth as vs
import vsutil
import vardefunc
import lvsfunc
import dysfunctional

from nnedi3_rpow2 import * 
from havsfunc import FineDehalo

core = vs.core
    
OPS, EDS = 3453, 31553
OP_HALOS = [(3462, 3529), (3585, 3633), (3678, 3725), (3867, 3930), (3983, 4040), (4084, 4150), (4235, 4297), (4334, 4393), (4454, 4515), (4552, 4611), (4665, 4718), (4770, 4841), (4882, 4929), (5038, 5109), (5152, 5217), (5261, 5338), (5375, 5490)]

src = core.lsmas.LWLibavSource()
src = vsutil.depth(src, 32)

descale = lvsfunc.scale.descale(src, width=1500, height=844, kernel=lvsfunc.kernels.Bilinear(), upscaler=None)

emask = vardefunc.mask.FreyChenG41().get_mask(descale).std.Inflate()

denoise = lvsfunc.util.quick_resample(descale, function=lambda x: dysfunctional.bm3dGPU(x, sigma=2, profile='lc'))
denoise = core.std.MaskedMerge(denoise, descale, emask)

deband = lvsfunc.util.quick_resample(denoise, function=lambda x: vardefunc.deband.dumb3kdb(x, preset='low/nograin'))
deband = core.std.MaskedMerge(deband, denoise, emask)
deband = dysfunctional.coolgrain(deband, strength=[3,0], radius=3, luma_scaling=10)

upscale = nnedi3_rpow2(deband, rfactor=2, width=1920, height=1080, kernel="spline64", upsizer='nnedi3cl')
upscale = core.std.ShufflePlanes([upscale, src], [0,1,2], vs.YUV)

op_halos = lvsfunc.rfs(src, FineDehalo(src, darkstr=0, brightstr=.6, rx=2, ry=3, edgeproc=2, showmask=0), ranges=OP_HALOS)
merge = core.std.MaskedMerge(upscale, op_halos, lvsfunc.scale.descale_detail_mask(src, upscale, threshold=0.05))

vsutil.depth(merge, 10).set_output()