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

src_path = Path("In.the.Heights.2021.2160p.UHD.Blu-ray.Remux.HEVC.HDR10P.DoVi.TrueHD.Atmos.7.1-WB4K.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top=276, bottom=276)

if filter:
    src = awf.fb(src, top=1, bottom=1)
    ssim = dysf.ssimdown(src, preset=1080, shader_str=shader)

    bicubic = awf.zresize(src, preset=1080, kernel='bicubic', filter_param_a=0, filter_param_b=0)
    rep = rgvs.Repair(ssim, bicubic, mode=20)

    src = core.std.Expr([ssim, rep], expr=[f'x y < x x y - {1} * - x x y - {0} * - ?'])
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()
