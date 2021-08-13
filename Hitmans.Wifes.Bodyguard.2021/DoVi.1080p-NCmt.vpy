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

src_path = Path("Hitmans.Wifes.Bodyguard.2021.2160p.UHD.Blu-ray.Remux.HEVC.HDR.DoVi.TrueHD.Atmos.7.1-LGF4K.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top = 274, bottom = 274)

if filter:
    src = awf.bbmod(src, top=1, bottom=1)
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()