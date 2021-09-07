import vapoursynth as vs
from vapoursynth import core

import awsmfunc
import vsutil

core.max_cache_size = 2048
core.num_threads = 8

# Source
src = core.ffms2.Source(r"C:\..\source.mkv")
src = core.std.Crop(src, top=20, bottom=20)
fltr = vsutil.depth(src, 16)
fill = awsmfunc.fb(fltr, bottom=1, top=1, mode="mirror")
fillc = awsmfunc.zresize(fill, 1080, top=1, bottom=1)
dirty = awsmfunc.bbmod(fillc, top=3, bottom=3, thresh=88, blur=55)
final = vsutil.depth(dirty, 8)

final.set_output()