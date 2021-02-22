import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional

from adptvgrnMod import adptvgrnMod

core.num_threads = 6
core.max_cache_size = 2048

src_path = Path("Source.E05.mkv")
src = core.ffms2.Source(src_path)
src = src.std.AssumeFPS(fpsnum = 25)
src = src.std.BlankClip(length = 1) + src + src.std.BlankClip(length = 3)

# 1080p
src = dysfunctional.ssimdown(src, preset=1080)
src = adptvgrnMod(src, strength=0.50, size=1.25, sharp=50, static=False, luma_scaling=15, grain_chroma=True, grainer=None, show_mask=False)

src.set_output()