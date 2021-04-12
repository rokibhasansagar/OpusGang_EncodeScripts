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

src_path = Path("Batman.v.Superman.Dawn.of.Justice.2016.Remastered.UHD.BluRay.2160p.TrueHD.Atmos.7.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)

if filter:
    fix_tb = src.std.Crop(top=276, bottom=276)
    fix_tb = awf.bbmod(fix_tb, top=2, bottom=2)
    fix_tb = dysf.ssimdown(fix_tb, preset = 1080)
    fix_tb = fix_tb.std.AddBorders(top=138, bottom=138)

    fix_lr = src.std.Crop(left=380, right=380)
    fix_lr = awf.bbmod(fix_lr, top=2, bottom=2, left=3, right=3)
    fix_lr = dysf.ssimdown(fix_lr, preset = 1080)
    fix_lr = fix_lr.std.AddBorders(left=190, right=190)

    src = dysf.ssimdown(src, preset = 1080)

    src = awf.rfs(src, fix_tb, mappings="[7388 93277] [99705 176895] [187531 231940] [245391 249553]")
    src = awf.rfs(src, fix_lr, mappings="[0 7387] [93278 99704] [176896 187530] [231941 245390] [249554 251357]")
else:
    src = dysf.ssimdown(src, preset = 1080)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()