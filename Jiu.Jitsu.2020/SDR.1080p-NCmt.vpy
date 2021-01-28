import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import adptvgrnMod
import rekt

core.num_threads = 4
core.max_cache_size = 4096

src_path = Path("Jiu.Jitsu.2020.1080p.BluRay.Remux.AVC.DTS-HD.MA.5.1-PmP.mkv")
src = core.ffms2.Source(src_path)

src = src.std.Crop(left = 0, right = 0, top = 136, bottom = 136)

if False:
    src = awf.fb(src, top=1, bottom=1)
    src = awf.bbmod(src, top=2, left=1, bottom=1)

    src = vsutil.depth(src, 16)

    mask = awf.bandmask(src, thr=250)

    debanded = core.neo_f3kdb.Deband(src, y=48, cb=48, cr=48, grainy=32, grainc=32, dynamic_grain=True, range=30, output_depth=16)
    debanded = adptvgrnMod.adptvgrnMod(debanded, strength=1.00, size=1.25, sharp=50, static=False, luma_scaling=25,grain_chroma=False, grainer=None, show_mask=False)

    src = core.std.MaskedMerge(src, debanded, mask)
    src = vsutil.depth(src, 8, dither_type='error_diffusion')

src.set_output()