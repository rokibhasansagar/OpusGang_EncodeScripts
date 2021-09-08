import vapoursynth as vs
from vapoursynth import core

from vsutil import depth, get_y
import awsmfunc as awf
from rekt import rekt
from adptvgrnMod import adptvgrnMod


def f32kdb(clip, range=15, y=32, cb=0, cr=0, sample_mode=4, dither="none"):
    """
    just an easier to read version of vardefunc's dumb3kdb with minor changes:
    * grain is always 0
    * changed defaults
    * clips are merged at 32-bit
    * you can also just pass a 32-bit clip (Idc if this is slower or something)
    """
    from vsutil import depth
    # 16 for sample_mode = 2
    # 32 for rest
    step = 16 if sample_mode == 2 else 32

    odepth = max(clip.format.bits_per_sample, 16)
    clip = depth(clip, 16, dither_type="none")

    if y % step == cb % step == cr % step == 0:
        return depth(clip.neo_f3kdb.Deband(range, y, cb, cr, 0, 0, sample_mode), odepth, dither_type=dither)
    else:
        loy, locb, locr = [max((th - 1) // step * step + 1, 0) for th in [y, cb, cr]]
        hiy, hicb, hicr = [min(lo + step, 511) for lo in [loy, locb, locr]]

        lo_clip = depth(clip.neo_f3kdb.Deband(range, loy, locb, locr, 0, 0, sample_mode), 32)
        hi_clip = depth(clip.neo_f3kdb.Deband(range, hiy, hicb, hicr, 0, 0, sample_mode), 32)

        if clip.format.color_family == vs.GRAY:
            weight = (y - loy) / step
        else:
            weight = [(y - loy) / step, (cb - locb) / step, (cr - locr) / step]

        return depth(core.std.Merge(lo_clip, hi_clip, weight), odepth, dither_type=dither)


core.num_threads = 8
filter = False

src = core.ffms2.Source("Lady.Vengeance.2005.UHD.BluRay.2160p.DTS-HD.MA.5.1.DV.HEVC.REMUX-FraMeSToR.mkv")

# top and bottom have 1px of pretty garbage info
crop = 264
src = src.std.Crop(top=crop, bottom=crop)

src = depth(src, 16)

if filter:
    src = rekt(src, awf.fb(src, top=1), left=3838, bottom=1630)
    src = rekt(src, awf.fb(src, bottom=1), left=3838, top=1630)

    src = depth(src, 32)

    src = rekt(src, awf.bbmod(src, top=7, right=6, blur=20), left=3810, bottom=1620)
    src = rekt(src, awf.bbmod(src, bottom=7, right=6, blur=20), left=3810, top=1620)

    # the dirty lines here are very inconsistent and seem more like edges of the film
    # they're way too much work to "fix" even when that's possible so this is just a lazy way to make them less of an eyesore

    src = awf.bbmod(src, left=6, right=1, blur=999, thresh=30)

    src = awf.zr(src, 1080, kernel='bicubic', top=1, bottom=1, filter_param_a=1/3, filter_param_b=1/3)

    dbn = f32kdb(src, y=40, cb=40, cr=40)
    msk = get_y(src).std.Sobel()
    mrg = core.std.MaskedMerge(dbn, src, msk.std.Binarize(0.015).std.Minimum().std.Maximum().std.Maximum())

    grn = adptvgrnMod(mrg, strength=.1, size=1.8, sharp=70, luma_scaling=5)
    mrg = core.std.MaskedMerge(grn, mrg, msk.std.Binarize(0.1).std.Maximum())
    src = awf.rfs(src, mrg, "[27859 27900] [50275 50620] [97364 103203] [127223 127252]")
else:
    src = awf.zr(src, 1080, kernel='bicubic', top=1, bottom=1, filter_param_a=1/3, filter_param_b=1/3)

src = depth(src, 10)

src.set_output()
