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
src = src.resize.Spline36(format=vs.YUV444P16)

src = src.std.Crop(top=67, bottom=67)

if filter:
    imax = awf.bbmod(src, top=2, bottom=2)
    imax = imax.resize.Spline36(width=1920, height=1013)

    reg = src.std.Crop(top=210, bottom=210)
    reg = awf.bbmod(reg, top=1, bottom=1)
    reg = reg.resize.Spline36(width=1920, height=803)
    reg = reg.std.AddBorders(top=105, bottom=105)

    src = src.resize.Spline36(width=1920, height=1013)

    src = awf.rfs(src, imax, mappings="[40 15917] [205064 209440]")
    src = awf.rfs(src, reg, mappings="[0 39] [16133 205063] [209441 217407]")

    src = src.resize.Lanczos(width=1920, height=1014)

    lumafix = rekt.rektlvls(src, rownum=[105, 106, 907, 908], rowval=[9, -3, -3, 9])
    chromafix = awf.bbmod(lumafix, top=105, bottom=105, y=False)
    fix = rekt.rekt(lumafix, chromafix, top=104, bottom=909)
    fix = rekt.rekt(fix, chromafix, top=909, bottom=104)

    src = awf.rfs(src, fix, mappings="[0 39] [16133 205063] [209441 217407]")

    src = src.resize.Spline36(format=vs.YUV420P16)
else:
    src = src.resize.Spline36(width=1920, height=1014)
    src = src.resize.Spline36(format=vs.YUV420P16)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()