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

core.num_threads = 8

filter = True

src_path = Path("Abominable.2019.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

src = src.std.Crop(top = 42, bottom = 42)
src = dysf.ssimdown(src, preset = 1080)

if filter:
    src = adptvgrnMod(src, strength=0.50, size=0.60, sharp=50, static=False, luma_scaling=80, grain_chroma=False, grainer=None, show_mask=False)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()