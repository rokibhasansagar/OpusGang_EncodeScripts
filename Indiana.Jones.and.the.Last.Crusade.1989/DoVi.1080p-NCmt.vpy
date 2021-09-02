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
import muvsfunc
import rgvs

from adptvgrnMod import adptvgrnMod

with open(Path("~/Encoding/ssimdown.glsl").expanduser()) as s:
    shader = s.read()

core.num_threads = 8
filter = True

src_path = Path("Indiana.Jones.and.the.Last.Crusade.1989.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top=262, bottom=262)

if filter:
    src = awf.fb(src, top=1, bottom=1)
    src = awf.bbmod(src, top=2, bottom=2)

    src = dysf.ssimdown(src, preset=1080, shader_str=shader, repair=[1, 0])
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)


src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()
