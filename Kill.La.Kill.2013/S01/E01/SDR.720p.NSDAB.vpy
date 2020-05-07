import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import havsfunc as haf
from adptvgrnMod import *
from functools import partial

def pancrop(clip, start, end, alt_clip):
    def staccrop(clip, value):
        left = clip.std.Crop(right=value).cf.ContinuityFixer(right=1, radius=5)
        right = clip.std.Crop(left=clip.width - value).cf.ContinuityFixer(left=1, radius=5)
        left = fvf.Debicubic(left, 1280 / 1920 * (1920 - value), 720, b=0, c=1, yuv444=True)
        right = fvf.Debicubic(right, 1280 / 1920 * value, 720, b=0, c=1, yuv444=True)
        return core.std.StackHorizontal([left, right])
    def cropeval(n, clip):
        if n == start:
            return staccrop(clip, 1536)
        if n == start + 1:
            return staccrop(clip, 1152)
        if n == start + 2:
            return staccrop(clip, 768)
        if n == start + 3:
            return staccrop(clip, 384)
        elif n < start or n > end:
            return fvf.Debicubic(clip, 1280, 720, b=0, c=1, yuv444=True)
            
    replaced = core.std.FrameEval(clip.resize.Point(1280, 720, format=vs.YUV444P16), partial(cropeval, clip=clip))
    return core.std.Trim(alt_clip, 0, start - 1) + core.std.Trim(replaced, start, end) + core.std.Trim(alt_clip, end + 1)

src = core.ffms2.Source("")

src = fvf.Depth(src, 16)

dns = fvf.Debicubic(src, 1280, 720, b=0, c=1, yuv444=True)
mds = fvf.DebicubicM(src, 1280, 720, b=0, c=1, yuv444=True)
dns = fvf.rfs(dns, mds, "[32726 34791]")
dns = pancrop(src, 35087, 35090, dns)
dns = pancrop(src, 35076, 35079, dns)
dns = pancrop(src, 35097, 35100, dns)
spl = core.resize.Spline36(src, 1280, 720, format=vs.YUV444P16)
dns = fvf.rfs(dns, spl, "[34837 34857] [34868 34883] [34907 34917] [34930 34951] [34964 34975] [35008 35021] [35037 35048] [35111 35124] [35140 35196]")

grn = core.grain.Add(dns, var=3, constant=False)
msk = core.std.Expr([dns, grn], "x 5000 < y 4096 - abs 0.7 * 4096 + x ?")
snf = fvf.rfs(dns, msk, "[4056 4089] [4769 4799] [20902 21013] [21066 21173] [21433 21524] [21555 21638] [25826 25857]")

grn = adptvgrnMod(dns, luma_scaling=3)
snf = fvf.rfs(snf, grn, "[9716 9877] [14741 14811] ")

drn = haf.HQDeringmod(dns)
grn = adptvgrnMod(grn, luma_scaling=3, static=False, strength=2)
snf = fvf.rfs(snf, grn, "[30840 30987]")

out = fvf.Depth(snf, 10)

out.set_output()