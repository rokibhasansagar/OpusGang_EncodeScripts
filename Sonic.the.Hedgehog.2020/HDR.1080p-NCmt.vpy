import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt

core.num_threads = 4
core.max_cache_size = 2048

src_path = Path("Sonic.the.Hedgehog.2020.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = src.std.Crop(left = 0, right = 0, top = 276, bottom = 276)
src = src.resize.Spline36(width = 1920, height = 804, dither_type = "error_diffusion")

src.set_output()