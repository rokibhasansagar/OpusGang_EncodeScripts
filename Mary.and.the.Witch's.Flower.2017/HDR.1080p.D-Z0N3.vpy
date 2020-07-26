import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
from vsutil import *
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from adptvgrnMod import *

# The film grain here is impossible to emulate properly with current methods, so I didn't fix the dynamic/static grain switching.

src = core.ffms2.Source("Mary and the Witch's Flower 2017 2160p UHD Blu-ray Remux v5 Dual Audio HEVC HDR10 DTS-HD MA 5 1.mkv")
src = src.std.Crop(top=42, bottom=42)
src = depth(src, 16)

# The whole thing is animated at 2560x1384.  I descaled and then resized to 1080p, because standards. :(
# Chroma had to be shifted by slightly less than -0.5.  I think this is how shifting works with double resizing.

y, u, v = split(src)
alt = [y.std.Crop(top=1, bottom=1), u, v]
y = fvf.Debicubic(y, 2560, 1384, b=0, c=.5)
y = awf.cr(y, 1080)
shift = .25 - .25 * 3840 / 2560 - .25 * 2560 / 1920
u, v = u.resize.Spline36(960, 519, src_left=shift), v.resize.Spline36(960, 519, src_left=shift)
src = join([y, u, v])

# Debanded luma.

dbn = y.placebo.Deband(threshold=4., radius=20., grain=8)
ems = kgf.kirsch(y).std.Maximum()
dbn = dbn.std.MaskedMerge(y, ems)
dbn = join([dbn, u, v])
src = awf.rfs(src, dbn, "[111752 111889]")

# Debanded intro luma and resized with spline.  I don't know why I manually shifted.

alt[0] = alt[0].resize.Spline36(src.width, src.height)
dbn = alt[0].placebo.Deband(threshold=6., radius=20., grain=0)
ems = kgf.kirsch(alt[0]).std.Maximum()
dbn = dbn.std.MaskedMerge(alt[0], ems)
alt[1], alt[2] = alt[1].resize.Spline36(u.width, u.height, src_left=-.25), alt[2].resize.Spline36(u.width, u.height, src_left=-.25)
dbn = join(alt)
dbn = sizedgrn(dbn, strength=.1, static=True)
src = awf.rfs(src, dbn, "[50 324]")

src = depth(src, 10)

src.set_output()
