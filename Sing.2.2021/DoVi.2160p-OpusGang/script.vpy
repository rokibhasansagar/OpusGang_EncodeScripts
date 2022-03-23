import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf

core.num_threads = 8
filter = True

src_path = Path(
    "Sing.2.2021.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv"
)
src = core.ffms2.Source(src_path)

# Dolby Vision FEL reconstruction
if filter:
    el_path = Path("EL.mkv")
    el = core.ffms2.Source(el_path)

    src = awf.MapDolbyVision(src, el)
    src = vsutil.depth(src, 16)

src = src.std.Crop(top=40, bottom=40)

if filter:
    src = awf.fb(src, top=1, bottom=1)
    src = awf.bbmod(src, top=2, bottom=2)

src = vsutil.depth(src, 10, dither_type="error_diffusion")

src.set_output()
