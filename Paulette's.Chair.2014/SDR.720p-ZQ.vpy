import awsmfunc
import vapoursynth as vs
import vsutil
import lvsfunc

from muvsfunc import firniture
from awsmfunc import bbmod
from rekt import rekt_fast
from havsfunc import FineDehalo
from dysfunctional import coolgrain
from adptvgrnMod import *
from G41Fun_PR import SuperToon
 
core = vs.core

source = core.ffms2.Source('')
b16 = vsutil.depth(source, 16)

def dehalo(clip: vs.VideoNode, mask: list = [16, 235, 2], blur: float = 1.3, show_mask: bool = False, **kwargs) -> vs.VideoNode:
    # Saw this and thought it was interesting
    # https://github.com/LightArrowsEXE/Encoding-Projects/blob/c1b2d56e2de4ef1fd4cb2e8f7c26deea8bfa7cd5/Miscellaneous%20groups/%5BFoxtrot%5D/Work%20in%20Progress/GREAT%20PRETENDER%20%5BBD%5D/GPBD_01.vpy#L69
    #Any, rad: int = 2, sigma: float = 1, brz: float = 0.35, thmi: int = 80, thma: int = 128, thlimi: int = 50, thlima: int = 100
    if isinstance(mask, int): div=[mask, mask, mask]

    y = vsutil.get_y(clip)
    pre_m = core.std.Limiter(y, min=mask[0] << 8, max=mask[1] << 8)
    mask = lvsfunc.mask.halo_mask(pre_m, rad=mask[2], **kwargs)

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
halos_2 = lvsfunc.rfs(halos_1, FineDehalo(halos_1, brightstr=0, rx=2), ranges=[(2615, 2712), (4704, 4881)])
halos_3 = lvsfunc.rfs(halos_2, SuperToon(dehalo(b16, mask=[20, 235, 1], show_mask=False, brz=1.0, thmi=0, thma=110), power=.2), ranges=[(144, 215)])

aa_msk_pre = core.std.Limiter(halos_3, min=50 << 8, max=200 << 8)
aa_mask = core.std.Prewitt(core.bilateral.Bilateral(aa_msk_pre, sigmaS=1, sigmaR=0.10), 0, 3)
aa_mask = vsutil.iterate(aa_mask, partial(core.std.Maximum), 3)

# lazy
aa = core.std.MaskedMerge(halos_3, lvsfunc.aa.upscaled_sraa(halos_3, rfactor=1.4), aa_mask)
aa = lvsfunc.rfs(halos_3, aa, ranges=[(28, 106), (2757, 2903), (3133, 3197), (3277, 3547), (4297, 4364), (4479, 4516), (4537, 4704), (4787, 5020)])

down = lvsfunc.rfs(firniture(aa, width=1280, height=720, kernel='binomial5', sigmoid=True, gamma=True), firniture(b16, width=1280, height=720, kernel='noalias4', sigmoid=True, gamma=True), ranges=[(5040, 5189)])

deband = core.f3kdb.Deband(down, preset='low/nograin')
deband = core.std.MaskedMerge(deband, down, core.std.Prewitt(down, 0, 5).std.Maximum())
deband = lvsfunc.rfs(down, deband, ranges=[(1484, 1514), (2890, 2926)])

#grain = coolgrain(deband, strength=[1,.5], luma_scaling=14)
grain = adptvgrnMod(deband, strength=.25, luma_scaling=14, seed=444, static=True)
vsutil.depth(grain, 8, dither_type='error_diffusion').set_output()

# fades
# --zones 216,229,crf=7/1464,1515,crf=8/2853,2985,crf=8/3557,3602,crf=8


