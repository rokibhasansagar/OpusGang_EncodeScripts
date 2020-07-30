import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt

core.num_threads = 4
core.max_cache_size = 2048

src_path = Path("Swallow.2019.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-NCmt.mkv")
src = core.ffms2.Source(src_path)

src = src.std.Crop(left = 0, right = 0, top = 138, bottom = 138)

src.set_output()