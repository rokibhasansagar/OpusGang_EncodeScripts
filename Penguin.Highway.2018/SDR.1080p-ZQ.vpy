import lvsfunc
import vapoursynth as vs
import vsutil
import dysfunctional

from vardefunc import dumb3kdb
from kagefunc import adaptive_grain, retinex_edgemask
from adptvgrnMod import *
from havsfunc import FineDehalo, scale
from awsmfunc import bbmod
from zzfunc import minfilter

core = vs.core

src = core.ffms2.Source('')
src = core.std.Crop(src, top=104, bottom=104)
b16 = vsutil.depth(src, 16)

denoise = dysfunctional.bm3dGPU(b16, sigma=2, radius=3)
denoise = core.knlm.KNLMeansCL(denoise, h=.6, a=3, channels='uv')

border1 = bbmod(denoise, top=3, bottom=3, left=3, right=3, planes=[0,1,2], blur=999, csize=6, thresh=6, scale_thresh=True) 
border2 = core.cf.ContinuityFixer(denoise, top=[3,3,3], bottom=[3,3,3], left=[3,3,3], right=[3,3,3], radius=[200,200,200])
borders = minfilter(denoise, border1, border2, planes=[0,1,2], strict=False)

deband = dumb3kdb(borders, radius=25, threshold=[30, 15, 15])
merge = core.std.MaskedMerge(deband, b16, dysfunctional.FDOG(b16))

halos = FineDehalo(merge, darkstr=0, rx=2, ry=2, thmi=25, thma=200, thlimi=100, thlima=128, showmask=0)
grain = adptvgrnMod(halos, strength=0.3, cstrength=0, static=True)

src.set_output(1)
borders.set_output(2)
merge.set_output(3)
grain.set_output(4)