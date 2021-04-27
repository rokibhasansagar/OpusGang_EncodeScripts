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

core.num_threads = 8

filter = True

src_path = Path("Wonder.Woman.1984.2020.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top=66, bottom=66)

if filter:
    imax = awf.fb(src, top=1, bottom=1)
    imax = awf.bbmod(imax, top=2, bottom=2)
    imax = dysf.ssimdown(imax, preset = 1080)

    reg = src.std.Crop(top=212, bottom=212)
    reg = awf.bbmod(reg, top=1, bottom=1)
    reg = dysf.ssimdown(reg, preset = 1080)
    reg = reg.std.AddBorders(top=106, bottom=106)

    src = dysf.ssimdown(src, preset = 1080)

    src = awf.rfs(src, imax, mappings="[40 15917] [205064 209440]")
    src = awf.rfs(src, reg, mappings="[0 39] [16133 205063] [209441 217407]")
else:
    src = dysf.ssimdown(src, preset = 1080)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()