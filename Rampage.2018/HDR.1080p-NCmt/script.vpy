import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional

core.num_threads = 4
core.max_cache_size = 2048

src_path = Path("Rampage.2018.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = src.std.Crop(left = 0, right = 0, top = 280, bottom = 280)
src = awf.fb(src, right=1)
src = awf.bbmod(src, right=2)
src = dysfunctional.ssimdown(src, preset=1080)

if False:
    src = vsutil.depth(src, 16)
    msk = awf.bandmask(src, thr=100)

    # Banding
    src = awf.DebandReader(src, "banding.txt", grain=40, mask=msk)

    src = vsutil.depth(src, 10)

src.set_output()