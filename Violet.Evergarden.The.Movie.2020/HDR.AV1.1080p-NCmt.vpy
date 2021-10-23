from vapoursynth import core, VideoNode, GRAY, YUV
import vapoursynth as vs
from pathlib import Path

import vsutil
import awsmfunc as awf
import dysfunctional as dysf
import kagefunc as kgf
import havsfunc as haf

import atomchtools as atf
import cooldegrain
import finedehalo as fd

with open(Path("~/Encoding/ssimdown.glsl").expanduser()) as s:
    shader = s.read()

core.num_threads = 12
filter = True

src_path = Path(
    "Violet.Evergarden.The.Movie.2020.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
src = src.std.Crop(top=274, bottom=274)
src = awf.fb(src, top=1, bottom=1)

if filter:
    pf = core.dfttest.DFTTest(src, sigma=1, tbsize=3, opt=3, planes=[0, 1, 2])
    src = cooldegrain.CoolDegrain(
        src, tr=1, thsad=8, thsadc=4, bits=16, blksize=8, overlap=4, pf=pf)

    src = dysf.ssimdown(src, preset=1080, shader_str=shader, repair=[1, 0])
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type='error_diffusion')
# src = core.std.Trim(src, 0, 50025)
# src = core.std.Trim(src, 50026, 99907)
# src = core.std.Trim(src, 99908, 149833)
# src = core.std.Trim(src, 149834)

src.set_output()
