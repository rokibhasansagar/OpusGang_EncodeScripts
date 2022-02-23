from functools import partial
import vapoursynth as vs

core = vs.core

import havsfunc
import lvsfunc
import muvsfunc
import rgvs
import vsmask
import vsTAAmbk as taa
import vsutil

from awsmfunc import awsmfunc
from awsmfunc.awsmfunc import bbmod
from awsmfunc.detect import autoDeband
from dysfunctional import autodeblock, retinex
from havsfunc import LSFmod

from rekt import rektlvls

from hells_common import (frequency_merge, knl, naa,
                          texture_mask, vsdpir)


sraa = [(4207, 4255), (9609, 9724), (15576, 15614), (17125, 17154), (17503, 17598), (35200, 35242),
        (44773, 44862), (45247, 45363), (45520, 45645), (46125, 46196), (58107, 58178), (58860, 58920),
        (59005, 59100), (59398, 59520), (69857, 70015), (70928, 70966), (73177, 73482), (140877, 140982),
        (161823, 161988), (162799, 162848)]

src = core.lsmas.LWLibavSource('00000.m2ts')
src = core.std.Crop(src, bottom=2)
src = vsutil.depth(src, 16)

srcLuma = vsutil.get_y(src)
srcPre = rgvs.MinBlur(srcLuma, radius=1)
edgeKirsch = retinex(srcPre, vsmask.edge.Kirsch().get_mask(srcPre),
                     msrcp_dict=dict(op=5))

edgeFilterRekt = rektlvls(src, rownum=[0, 1, 1077, 1076], rowval=[15, -4, -7, -4],
                            colnum=[0, 1, 1918, 1919], colval=[15, -4, -7, 10])
edgesFilterBbAlt = bbmod(edgeFilterRekt, top=3, bottom=3, left=3, right=3,
                    blur=[40]*3, planes=[0, 1, 2], thresh=12, scale_thresh=True)

edgesFilter = muvsfunc.MaskedLimitFilter(edgesFilterBbAlt, edgeFilterRekt, thr=3, elast=5.0, planes=[0, 1, 2])

antiAWeak = naa(edgesFilter)
antiAStrong = taa.TAAmbk(edgesFilter, aatype='Eedi2', cuda=True, opencl=True)
antiAFuck = lvsfunc.aa.upscaled_sraa(edgesFilter)

antiA = lvsfunc.rfs(antiAWeak, antiAStrong, ranges=[(3211, 3211), (3814, 3823), (5240, 5386), (58701, 58821), (69857, 70015)])
antiA = lvsfunc.rfs(antiA, antiAFuck, ranges=sraa)
antiA = LSFmod(antiA, source=edgesFilter, defaults='fast')

haloMask = lvsfunc.util.quick_resample(antiA, lambda e: e.std.Expr(["x sqrt"]))
haloMask = lvsfunc.mask.halo_mask(haloMask, rad=1)
haloMask = vsmask.util.expand(haloMask, sw=1, sh=2)

haloFilter = havsfunc.DeHalo_alpha(antiA, brightstr=1, darkstr=0.8)
haloFilter = muvsfunc.MaskedLimitFilter(haloFilter, antiA, thr=5, elast=7.0)
haloFilter = muvsfunc.MaskedLimitFilter(havsfunc.Toon(haloFilter), haloFilter, ref=antiA, thr=3, elast=4)

haloMerge = core.std.MaskedMerge(antiA, haloFilter, haloMask)
haloFail = core.std.MaskedMerge(haloFilter, haloMerge, edgeKirsch)
haloMerge = lvsfunc.rfs(haloMerge, haloFail, ranges=[(35249, 35361), (64031, 64101), (66245, 66295), (69857, 70015), (71871, 72055)])

textureMask = core.std.Expr([texture_mask(src), edgeKirsch.std.Minimum()], ["x y + 65535 min"])
ringFilter = core.knlm.KNLMeansCL(haloMerge, h=1.5, a=3, s=2, d=1)
ringFilter = frequency_merge(haloMerge, ringFilter, thresh=16)
ringFilter = muvsfunc.MaskedLimitFilter(ringFilter, haloMerge, thr=2, elast=7.0)
ringMerge = core.std.MaskedMerge(ringFilter, haloMerge, textureMask)

deblockAuto = autodeblock(ringMerge, func=partial(knl),
                            strs=[2, 3, 4], write_props=True,
                            thrs=[(1, 15, 15), (2, 25, 25), (3, 35, 35)])
deblockManual = vsdpir(src, task='deblock', strength=50)
deblock = lvsfunc.rfs(deblockAuto, deblockManual, ranges=[(4598, 4681), (26382, 26442), (66989, 67075), (70945, 70966)])

debandAuto = autoDeband(deblock, thr=7, deband_range=(40, 80, 3), deband_scale=2,
                        downsample=720, cambi_args=dict(topk=0.001, tvi_threshold=0.01),
                        grainer=False, debug=False)
debandManual = core.neo_f3kdb.Deband(debandAuto, range=28, preset='nograin', sample_mode=4)
debandManual = muvsfunc.MaskedLimitFilter(debandManual, deblock, thr=2.5, elast=4)
deband = lvsfunc.rfs(debandAuto, debandManual, ranges=[(3530, 3597), (10777, 10923), (13571, 13667), (63554, 63720), (119518, 119617), (137073, 137254), (143770, 144046)])

ohGodOhFuck = rgvs.MinBlur(src, radius=4)
ohGodOhFuck = muvsfunc.BMAFilter(src, guidance=ohGodOhFuck, radius=1)
ohGodOhFuck = lvsfunc.rfs(deband, ohGodOhFuck, ranges=[(101502, 101545)])

#grain = adptvgrnMod(ohGodOhFuck, strength=0.08, grain_chroma=False,
#                    static=True, lo=18, seed=422)
out = core.fmtc.bitdepth((ohGodOhFuck[:166574] + src[166575:]), bits=10, dmode=9, tpdfo=1)
out = awsmfunc.SelectRangeEvery(out, 1500, 30, [5000, 10000])

out.set_output()