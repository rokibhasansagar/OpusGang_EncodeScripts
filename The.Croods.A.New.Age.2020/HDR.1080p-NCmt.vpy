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

src_path = Path("The.Croods.A.New.Age.2020.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = src.std.Crop(left = 0, right = 0, top = 262, bottom = 262)
src = awf.fb(src, top = 1, bottom = 3)
src = awf.bbmod(src, top = 2)
src = dysfunctional.ssimdown(src, preset=1080)
src = adptvgrnMod(src, strength=0.40, size=0.60, sharp=50, static=False, luma_scaling=80, grain_chroma=False, grainer=None, show_mask=False)

src.set_output()