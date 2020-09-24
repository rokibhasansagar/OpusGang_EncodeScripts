import vapoursynth as vs
core = vs.core
core.max_cache_size = 1024
from vsutil import *
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from adptvgrnMod import *

src = core.ffms2.Source("The.Lion.King.2019.2160p.WEB-DL.DDP5.1.Atmos.HEVC-BLUTONiUM.mkv")
src_= src
src = depth(src, 16)

y, u, v = split(src)
y = fvf.Debilinear(y, get_w(1152), 1152)
w, h = 1920, 1080
y = y.resize.Point(format=vs.YUV444P16)
y = y.placebo.Shader("SSimDownscaler.glsl", w, h, filter="mitchell", linearize=0)
y = get_y(y)
y_= y
shift = .25 - .25 * 3840 / get_w(1152) - .25 * get_w(1152) / w
u = u.resize.Spline36(960, 540, src_left=shift)
v = v.resize.Spline36(960, 540, src_left=shift)
src = join([y, u, v])
dbn = src.neo_f3kdb.Deband(range=20, y=48, cb=48, cr=48, grainy=0, grainc=0, sample_mode=4)
ems = kgf.kirsch(y_).std.Binarize(8000).std.Inflate()
dbn = dbn.std.MaskedMerge(src, ems)
dbn = sizedgrn(dbn, strength=.1, cstrength=0, static=True, protect_neutral=False, fade_edges=True, seed=32481)
bms = awf.bandmask(y_, thr=120, mid=2)
dbn = src.std.MaskedMerge(dbn, bms)
src = adptvgrnMod(dbn, strength=.2, cstrength=.08, luma_scaling=150, seed=768, show_mask=False)

src = depth(src, 8)
src = src.std.BlankClip(color=[16, 128, 128], length=14) + src

src.set_output()
