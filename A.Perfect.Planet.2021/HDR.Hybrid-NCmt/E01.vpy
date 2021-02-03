import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional

from adptvgrnMod import adptvgrnMod

core.num_threads = 4
core.max_cache_size = 4096

src_path = Path("Source.E01.mkv")
src = core.ffms2.Source(src_path)
src = src.std.AssumeFPS(fpsnum = 25)
src = src.std.BlankClip(length = 1) + src

extras_path = Path("a.perfect.planet.s01e01.hlg.2160p.web.h265-glhf.mkv")
extras = core.ffms2.Source(extras_path)
extras = extras.std.BlankClip(length = 150) + extras
extras = extras.resize.Spline36(transfer_in_s="std-b67", transfer_s="st2084")

"""
bd_path = Path("A.Perfect.Planet.S01E01.1080p.BluRay.x264-FLAWLESSWORLD.mkv")
bd = core.ffms2.Source(bd_path)
bd = vsutil.depth(bd, 16)
bd = awf.FrameInfo(bd, "BD")
"""

src = src[:76352] + src.std.BlankClip(length = 16) + extras[76368:86969] + src.std.BlankClip(length = 16) + src[76336:] + src.std.BlankClip(length = 1)

src = dysfunctional.ssimdown(src, preset=1080)

src.set_output()