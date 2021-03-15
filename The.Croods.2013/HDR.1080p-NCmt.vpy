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

filter = False

src_path = Path("The.Croods.2013.UHD.BluRay.2160p.DTS-X.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

src = src.std.Crop(top = 262, bottom = 262)
src = awf.fb(src, top = 1, bottom = 1)

if filter:
    src = awf.bbmod(src, top = 2, bottom = 2)
    src = src.resize.Spline36(width=1920, height=818)
    src = adptvgrnMod(src, strength=0.40, size=0.60, sharp=50, static=False, luma_scaling=80, grain_chroma=False, grainer=None, show_mask=False)
else:
    src = src.resize.Spline36(width=1920, height=818)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()