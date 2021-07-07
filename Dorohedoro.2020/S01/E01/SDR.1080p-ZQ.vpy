import vapoursynth as vs

import vsutil
import vardefunc
import lvsfunc
import dysfunctional
import dorohedoro
import debandshit

from rekt import rekt_fast

core = vs.core

# descale mask works just fine for ED
OPS, OPE = 3429, 5586

OP = core.lsmas.LWLibavSource()
OP = vsutil.depth(OP, 32)

src = core.lsmas.LWLibavSource()[24:-24]
src = vsutil.depth(src, 32)

descale = dorohedoro.rescale(src, width=1500, height=844, kernel=lvsfunc.kernels.Bilinear())

emask = vardefunc.mask.FreyChenG41().get_mask(descale, multi=1).std.Inflate()
denoise = core.std.MaskedMerge(dorohedoro.SM3D(descale), descale, emask)

dmask = vardefunc.mask.FDOG().get_mask(denoise)
dmask = dorohedoro.retinex(dmask, msrcp_dict=dict(fulls=True), tcanny_dict=dict(mode=1))

deband_light = vardefunc.placebo.deband(denoise, threshold=1, radius=24, grain=0, iterations=4)
deband_heavy = lvsfunc.util.quick_resample(denoise, function=lambda x: debandshit.f3kpf(x, radius=31,threshold=[84]*3))
deband = lvsfunc.rfs(deband_light, deband_heavy, ranges=[(1034, 1141), (26130, 26276)])
deband = core.std.MaskedMerge(deband, denoise, dmask, planes=[0,1,2])

aa_1 = rekt_fast(deband, fun=lambda x: dorohedoro.sraa(x, rfactor=1.3), top=deband.height - 850, left=deband.width - 620)
aa_1 = lvsfunc.rfs(deband, aa_1, ranges=(13061, 13107))

aa_2 = dorohedoro.sraa(aa_1, rfactor=1.5, planes=[0,1,2])
aa_2 = lvsfunc.rfs(aa_1, aa_2, ranges=([(7339, 7422), (7639, 7794), (11001, 11001), (11883, 12002), (14318, 14485), (19292, 19420), (23826, 24017)]))

grain = dysfunctional.coolgrain(aa_2, strength=[2,0], radius=2, luma_scaling=16, cutoff=16.5, divby=1.3, filter='robidoux')

opening = vsutil.insert_clip(grain, OP, start_frame=OPS)
vsutil.depth(opening, 10).set_output()