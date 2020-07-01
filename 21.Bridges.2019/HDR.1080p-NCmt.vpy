import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt

core.num_threads = 4
core.max_cache_size = 2048

src_path = Path("21.Bridges.2019.UHD.BluRay.2160p.DTS-HD.MA.5.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = src.std.Crop(left = 0, right = 0, top = 276, bottom = 276)
src = awf.bbmod(src, top=4, bottom=4)
src = src.resize.Spline36(width = 1920, height = 804, dither_type = "error_diffusion")

if False:
    s16 = vsutil.depth(src, 16)
    msk = awf.bandmask(s16, thr=175)

    # Banding
    s16 = awf.DebandReader(s16, "banding.txt", grain=40, mask=msk)

    src = vsutil.depth(s16, 10)

src.set_output()

# Banding.txt
""" 
599 1410 84
 """