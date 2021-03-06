import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional
import havsfunc
import vsTAAmbk
import kagefunc as kgf
import mvsfunc as mvf
import finesharp

from adptvgrnMod import adptvgrnMod

core.num_threads = 4
core.max_cache_size = 2048

filter = True

src_path = Path("Beauty.and.the.Beast.1991.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

if filter:
    # dehalo
    src = havsfunc.FineDehalo(src, contra=2.0, darkstr=0.60)

    # dering
    src = havsfunc.HQDeringmod(src, mthr=60, mrad=2)

    # some AA
    src = vsTAAmbk.TAAmbk(src, aatype=3, mtype=2, strength=1.0, cycle=2, mthr=10, mlthresh=24, opencl=False, down8=False)
    
    # get back some sharpness
    src = finesharp.sharpen(src, sstr=1.5)

    # resize
    src = dysfunctional.ssimdown(src, preset=1080)
else:
    src = dysfunctional.ssimdown(src, preset=1080)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()