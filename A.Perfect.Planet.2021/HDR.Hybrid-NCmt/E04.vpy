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

src_path = Path("Source.E04.mkv")
src = core.ffms2.Source(src_path)
src = src.std.AssumeFPS(fpsnum = 25)
src = src.std.BlankClip(length = 1) + src

extras_path = Path("a.perfect.planet.s01e04.hlg.2160p.web.h265-glhf.mkv")
extras = core.ffms2.Source(extras_path)
extras = extras.std.BlankClip(length = 150) + extras
extras = extras.resize.Spline36(transfer_in_s="std-b67", transfer_s="st2084")

"""
bd_path = Path("A.Perfect.Planet.S01E04.1080p.BluRay.x264-FLAWLESSWORLD.mkv")
bd = core.ffms2.Source(bd_path)
bd = vsutil.depth(bd, 16)
bd = awf.FrameInfo(bd, "BD")
"""

src = src[:71740] + src.std.BlankClip(length = 50) + extras[71790:85441] + src.std.BlankClip(length = 27) + src[71764:] + src.std.BlankClip(length = 4)

# 1080p
#src = dysfunctional.ssimdown(src, preset=1080)
#src = adptvgrnMod(src, strength=0.50, size=1.25, sharp=50, static=False, luma_scaling=15, grain_chroma=True, grainer=None, show_mask=False)

src.set_output()