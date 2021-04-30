import vapoursynth as vs
import vsutil
import lvsfunc

from muvsfunc import firniture
from awsmfunc import bbmod
from rekt import rekt_fast
from havsfunc import FineDehalo
from dysfunctional import coolgrain

core = vs.core

source = core.ffms2.Source('')
b16 = vsutil.depth(source, 16)

def dehalo(clip: vs.VideoNode, mask: list = [16, 235, 2], blur: float = 1.3, show_mask: bool = False) -> vs.VideoNode:
    # Saw this and thought it was interesting
    # https://github.com/LightArrowsEXE/Encoding-Projects/blob/c1b2d56e2de4ef1fd4cb2e8f7c26deea8bfa7cd5/Miscellaneous%20groups/%5BFoxtrot%5D/Work%20in%20Progress/GREAT%20PRETENDER%20%5BBD%5D/GPBD_01.vpy#L69

    if isinstance(mask, int): div=[mask, mask, mask]

    y = vsutil.get_y(clip)
    pre_m = core.std.Limiter(y, min=vsutil.scale_value(mask[0], 8, 16), max=vsutil.scale_value(mask[1], 8, 16))
    mask = lvsfunc.mask.halo_mask(pre_m, rad=mask[2])

    # probably just easier to use the other gauss plugins but where's the fun in that
    blur = core.placebo.Resample(y, width=clip.width, height=clip.height, filter='gaussian', blur=blur, param1=blur / 3, taper=blur / 2, radius=blur, lut_entries=128)
    dehalo = core.std.MaskedMerge(y, blur, mask)

    if show_mask is True:
        return mask
    else: return core.std.ShufflePlanes([dehalo, clip], [0,1,2], vs.YUV)

edges_1 = lvsfunc.rfs(bbmod(b16, top=2, bottom=2, left=2, right=2, planes=[0], blur=999), bbmod(b16, top=4, bottom=4, left=4, right=6, planes=[0,1,2], blur=45), ranges=[(1635, 1694)])
edges_2 = lvsfunc.rfs(edges_1, bbmod(b16, top=2, bottom=2, left=8, right=8, planes=[0,1,2], blur=20), ranges=[(2757, 2903)])

halos_1 = rekt_fast(edges_2, lambda x: dehalo(x, mask=[16, 144, 2], blur=1.5, show_mask=False), left=source.width - 300)
halos_1 = lvsfunc.rfs(edges_2, halos_1, ranges=[(2367, 2404)])

halos_2 = lvsfunc.rfs(halos_1, FineDehalo(halos_1, brightstr=0), ranges=[(2615, 2712)])

down = firniture(halos_2, width=1280, height=720, kernel='binomial5', sigmoid=True, gamma=True)

deband = core.f3kdb.Deband(down, preset='low/nograin')
deband = core.std.MaskedMerge(deband, down, core.std.Prewitt(down, 0, 5).std.Maximum())
deband = lvsfunc.rfs(down, deband, ranges=[(2890, 2926)])

grain = coolgrain(deband, strength=[1,.5], luma_scaling=14)
vsutil.depth(grain, 8).set_output()