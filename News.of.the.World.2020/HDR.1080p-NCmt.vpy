import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional as dysf
import vsTAAmbk
import kagefunc as kgf
import mvsfunc as mvf
import lvsfunc as lvf

from adptvgrnMod import adptvgrnMod

core.num_threads = 8

filter = True

src_path = Path("News.of.the.World.2020.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

src = src.std.Crop(top = 276, bottom = 276)
src = dysf.ssimdown(src, preset = 1080)

if filter:
    mask = awf.bandmask(src, thr=65)

    debanded = core.neo_f3kdb.Deband(src, y=32, cb=32, cr=32, grainy=10, grainc=10, dynamic_grain=True, range=30, output_depth=16)
    debanded = adptvgrnMod(debanded, strength=0.50, size=1.00, sharp=50, static=False, luma_scaling=60, grain_chroma=False, grainer=None, show_mask=False)

    debanded = core.std.MaskedMerge(src, debanded, mask)

    src = awf.rfs(src, debanded, mappings="[14546 14703] [122107 122157]")

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()