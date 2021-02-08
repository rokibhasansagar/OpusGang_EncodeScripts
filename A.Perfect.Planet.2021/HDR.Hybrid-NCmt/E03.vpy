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

src_path = Path("Source.E03.mkv")
src = core.ffms2.Source(src_path)
src = vsutil.depth(src, 16)
src = src.std.AssumeFPS(fpsnum = 25)
src = src.std.BlankClip(length = 1) + src
src = dysfunctional.ssimdown(src, preset=1080)
src = awf.DynamicTonemap(src, src_fmt=True, libplacebo=False, max_chroma=True, adjust_gamma=True)
src = awf.FrameInfo(src, "Source")

extras_path = Path("a.perfect.planet.s01e03.hlg.2160p.web.h265-glhf.mkv")
extras = core.ffms2.Source(extras_path)
extras = vsutil.depth(extras, 16)
extras = extras.std.BlankClip(length = 150) + extras
extras = extras.resize.Spline36(transfer_in_s="std-b67", transfer_s="st2084")
extras = dysfunctional.ssimdown(extras, preset=1080)
extras = awf.DynamicTonemap(extras, src_fmt=True, libplacebo=False, max_chroma=True, adjust_gamma=True)
extras = awf.FrameInfo(extras, "HLG")

bd_path = Path("A.Perfect.Planet.S01E03.1080p.BluRay.x264-FLAWLESSWORLD.mkv")
bd = core.ffms2.Source(bd_path)
bd = vsutil.depth(bd, 16)
bd = awf.FrameInfo(bd, "BD")

src = core.std.Interleave([src, bd, extras])
#src = src[:72756] + src.std.BlankClip(length = 27) + extras[72783:85337] + src.std.BlankClip(length = 38) + src[72811:] + src.std.BlankClip(length = 3)

# 1080p
#src = dysfunctional.ssimdown(src, preset=1080)
#src = adptvgrnMod(src, strength=0.50, size=1.25, sharp=50, static=False, luma_scaling=15, grain_chroma=True, grainer=None, show_mask=False)

src.set_output()