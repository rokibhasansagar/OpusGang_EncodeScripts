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

from adptvgrnMod import adptvgrnMod

with open(Path("~/Encoding/ssimdown.glsl").expanduser()) as s:
    shader = s.read()

core.num_threads = 8
do_it = True
filter = do_it
recon = do_it

src_path = Path(
    "Uncut.Gems.2019.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv"
)
src = core.ffms2.Source(src_path)
orig = src

src = vsutil.depth(src, 16)

# Dolby Vision FEL reconstruction
if recon:
    # Polynomial + MMR mapping
    src = core.placebo.Tonemap(
        src,
        src_csp=3,
        dst_csp=1,
    )

    # placebo.Tonemap returns YUV444P16, so go resize down
    src = core.resize.Spline36(src, format=vs.YUV420P16)

    # NLQ mapping with FEL
    # The results are better with both BL and EL in full range
    el_path = Path("EL.mkv")
    el = core.ffms2.Source(el_path)
    el = core.resize.Point(el, width=src.width, height=src.height)

    # Do the NLQ mapping
    src = core.vsnlq.MapNLQ(src, el)

# Final filtering & downscale
src = src.std.Crop(top=276, bottom=276)

# Filter & resize
if filter:
    src = awf.bbmod(src, top=4, bottom=4)
    src = dysf.ssimdown(src, preset=1080, shader_str=shader, repair=[1, 0])
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type="error_diffusion")

src.set_output()
