import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
from vsutil import *
import awsmfunc as awf
import havsfunc as haf
import lvsfunc as lvf
from adptvgrnMod import *

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
        return depth(clip.neo_f3kdb.Deband(range, y, cb, cr, 0, 0, sample_mode), odepth)
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

src = core.lsmas.LWLibavSource("Akira.1988.30th.Anniversary.Hybrid.1080p.BluRay.Remux.AVC.TrueHD.5.1-EDPH.mkv")

src = src.std.Crop(top=24, bottom=24)

src = depth(src, 32)

spl = split(src)

msk = lvf.mask.halo_mask(spl[0], rad=3, brz=.25, thmi=.18, thma=.2, thlimi=.2, thlima=.4).std.Limiter(0, 1)

ydh = haf.DeHalo_alpha(spl[0], rx=2.5, ry=2.5, brightstr=1, darkstr=0)

spl[0] = spl[0].std.MaskedMerge(ydh, msk)

dh = join(spl)

aw = lvf.util.quick_resample(dh, lambda x: x.warp.AWarpSharp2(blur=17, type=0, chroma=0, depth=32, planes=[1, 2]))

msk = spl[0].std.Prewitt().std.BinarizeMask(.5).std.Maximum().std.Maximum()
msk = core.std.Expr([msk, spl[0]], "y .23 < x 0 ?")
msk = msk.std.Maximum().std.Deflate()
msk = core.std.Expr([msk, spl[0]], "y .33 < x 0 ?")

mrg = core.std.MaskedMerge(dh, aw, msk)

src = core.std.Expr([dh, mrg], ["", "y abs x abs - 0 > x y ?"])

dbn = f32kdb(src, 15, 30, 40, 40)
dbn = adptvgrnMod(dbn, strength=.2, cstrength=.2, size=1.3)
src = awf.rfs(src, dbn, "[163790 163892] [163598 163673] [141964 142059] [173456 174109]")

v = [46, 46, 20, 18]
fix = src.std.Crop(v[0], v[1], v[2], v[3])
fix = awf.bbmod(fix, left=3, top=3, right=4, bottom=5, blur=999)
fix = fix.std.AddBorders(v[0], v[1], v[2], v[3])
src = awf.rfs(src, fix, "[173456 173715]")

v[3] += 4
fix = src.std.Crop(v[0], v[1], v[2], v[3])
fix = awf.bbmod(fix, left=3, top=3, right=4, bottom=0, blur=999)
fix = lvf.util.quick_resample(fix, lambda x: x.cf.ContinuityFixer(bottom=4, radius=10))
fix = fix.std.AddBorders(v[0], v[1], v[2], v[3])
src = awf.rfs(src, fix, "[173716 173828]")

fix = src.std.Crop(v[0], v[1], v[2], v[3])
fix = awf.bbmod(fix, left=3, top=3, right=4, bottom=0, blur=999)
fix = fix.std.AddBorders(v[0], v[1], v[2], v[3])
src = awf.rfs(src, fix, "[173829 179039]")

src = depth(src, 8)

src.set_output()