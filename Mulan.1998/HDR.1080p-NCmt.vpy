import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional
import havsfunc
import vsTAAmbk

from adptvgrnMod import adptvgrnMod

core.num_threads = 4
core.max_cache_size = 2048

filter = True

src_path = Path("Mulan.1998.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

src = src.std.Crop(left = 120, right = 120, top = 0, bottom = 0)
src = dysfunctional.ssimdown(src, preset=1080)

if filter:
    mask = dysfunctional.FDOG(src, div=2, sigma=1.0)
    #mask = awf.bandmask(src, thr=150)

    debanded = core.neo_f3kdb.Deband(src, y=40, cb=40, cr=40, grainy=10, grainc=10, dynamic_grain=True, range=20, output_depth=16)
    debanded = adptvgrnMod(debanded, strength=0.40, size=0.60, sharp=50, static=False, luma_scaling=80, grain_chroma=False, grainer=None, show_mask=False)

    src = core.std.MaskedMerge(debanded, src, mask)
    src = havsfunc.FineDehalo(src, contra=2.0, thmi=50, thma=85, thlimi=40, thlima=80, darkstr=0.40)
    src = havsfunc.HQDeringmod(src, mthr=35, thr=10.0, darkthr=10.0/6.0)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()