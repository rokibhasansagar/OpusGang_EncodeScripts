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
filter = True

src_path = Path("Looper.2012.2160p.UHD.Blu-ray.Remux.HEVC.HDR10P.DTS-HD.MA.5.1-LOOP.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top = 276, bottom = 276)

if filter:
    src = awf.fb(src, top=1)
    src = awf.bbmod(src, top=2, bottom=2)

    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

    src = adptvgrnMod(src, strength=2.75, size=0.95, sharp=45, static=False, luma_scaling=45, grain_chroma=False, grainer=None, show_mask=False)
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()
