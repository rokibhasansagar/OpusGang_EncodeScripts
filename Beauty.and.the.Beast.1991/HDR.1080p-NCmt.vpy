import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional as dysf
import vsTAAmbk
import kagefunc as kgf
import mvsfunc as mvf

from adptvgrnMod import adptvgrnMod

core.num_threads = 6

filter = True

src_path = Path("Beauty.and.the.Beast.1991.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

if filter:
    # resize
    src = src.resize.Spline36(width=1920, height=1080, dither_type = "error_diffusion")

    # dering
    src = haf.HQDeringmod(src, mthr=5, mrad=1, thr=4, sharp=2, darkthr=4/8)

    # dehalo
    src = haf.FineDehalo(src, contra=4.0, darkstr=0.60)

    # some AA
    src = vsTAAmbk.TAAmbk(src, aatype=3, mtype=2, strength=0.20, cycle=1, mthr=2, mlthresh=6, opencl=False, down8=False, sharp=2.0, aarepair=1)
else:
    src = dysf.ssimdown(src, preset=1080)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()