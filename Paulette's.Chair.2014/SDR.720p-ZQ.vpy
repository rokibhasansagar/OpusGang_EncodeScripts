import vapoursynth as vs
import vsutil
import lvsfunc

from muvsfunc import firniture
from awsmfunc import bbmod
from rekt import rekt_fast

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

edges = lvsfunc.rfs(bbmod(b16, top=2, bottom=2, left=2, right=2, planes=[0], blur=999), bbmod(b16, top=4, bottom=4, left=4, right=6, planes=[0,1,2], blur=45), ranges=[(1635, 1694)])

ew = rekt_fast(edges, lambda x: dehalo(x, mask=[16, 144, 2], blur=1.5, show_mask=False), left=edges.width - 300)
ew = lvsfunc.rfs(edges, ew, ranges=[(2404, 2404)])

down = firniture(ew, 1280, 720, kernel='binomial5', sigmoid=True, gamma=True)

vsutil.depth(down, 8).set_output()