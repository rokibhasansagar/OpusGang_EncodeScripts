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
import lvsfunc as lvf

from adptvgrnMod import adptvgrnMod

with open(Path("~/Encoding/ssimdown.glsl").expanduser()) as s:
    shader = s.read()

core.num_threads = 8
filter = False

src_path = Path("Raya.and.the.Last.Dragon.2021.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top = 276, bottom = 276)

if filter:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)
    src = adptvgrnMod(src, strength=0.40, size=0.60, sharp=50, static=False, luma_scaling=80, grain_chroma=False, grainer=None, show_mask=False)
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()