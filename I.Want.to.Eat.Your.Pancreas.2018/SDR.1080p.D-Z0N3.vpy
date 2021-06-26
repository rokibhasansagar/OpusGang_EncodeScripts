import vapoursynth as vs
core = vs.core
from vsutil import *
import fvsfunc as fvf
from rekt import *
import awsmfunc as awf
from edi_rpow2 import *
import havsfunc as haf
core.max_cache_size = 1024

src = core.ffms2.Source("remux.mkv")

src = depth(src, 32)

# dirty lines
src = rektlvls(src, colnum=[0, 1919], colval=2 * [19], rownum=[0, 1079], rowval=2 * [19], prot_val=[16, 235])
src = awf.bbmod(src, 1, 1, 1, 1, blur=999, thresh=9, planes=0)

y = get_y(src)

dsc = fvf.Debicubic(y, 1280, 720, b=0, c=.5)
ups_zn = znedi3_rpow2(dsc, 2).resize.Spline36(1920, 1080)
shader = "../Shaders/FSRCNNX_x2_56-16-4-1-NEW.glsl"
ups = dsc.resize.Point(format=vs.YUV444P16).placebo.Shader(shader=shader, width=dsc.width * 2, height=dsc.height * 2, filter="ewa_lanczos").resize.Spline36(1920, 1080, format=vs.GRAYS, matrix_s="709")
ups = core.std.Expr([ups, ups_zn], "x y min")

# descale and edge mask for rescale
msk = core.std.Expr([y, dsc.resize.Bicubic(1920, 1080)], "x y -")
msk = msk.std.Binarize(.05)
msk = iterate(msk, core.std.Maximum, 2)
msk = iterate(msk, core.std.Inflate, 2)
msk = [msk, y.std.Sobel()]
msk[1] = msk[1].std.Minimum().std.Maximum().std.Binarize(.05).std.Inflate().std.Inflate()

msk = core.std.Expr(msk, "y x -")

rsc = y.std.MaskedMerge(ups, msk)

y = awf.rfs(rsc, y, "[0 265] [145674 152155] [155383 155760]")

u = plane(src, 1)
v = plane(src, 2)

# black lines have green and purple haloing on chroma
def fixchroma(c, msk, luma, u, v):
    # only want to work on dark luma and its surroundings
    luma = luma.resize.Bicubic(c.width, c.height, filter_param_a=1, filter_param_b=0)
    # darken darks so we can use as reference
    luma = luma.std.Levels(gamma=.5)
    luma = luma.std.Minimum()
    # desaturate by deflating
    d = c.std.Expr("x abs")
    m = c.std.Inflate()
    d = d.std.Deflate()
    # limit values so high saturation can't be destroyed
    lim_lo = scale_value(117, 8, 32, chroma=True, scale_offsets=True)
    lim_hi = scale_value(144, 8, 32, chroma=True, scale_offsets=True)
    # maximum change possible
    thr = abs(scale_value(10, 8, 32, scale_offsets=True))
    # if it's low enough saturation and within the limits do d else c
    expr = f"y abs .08 < x {lim_lo} > x {lim_hi} < and and x 0 < y -1 * y ? x ?"
    # use u and v to filter out other colors and limit max change to thr
    d = core.std.Expr([c, d, luma, v, u, m], f"a 0.05 > b 0 < and x z 0.5 < x {expr} - {thr} > x {thr} - x {expr} - -{thr} < x {thr} + {expr} ? ? x ? ?")
    # this is probably stupid    
    d = haf.DeHalo_alpha(d, rx=1, ry=1, darkstr=2, brightstr=2, ss=2, lowsens=50, highsens=50)
    # limit to edges obviously
    msk = msk.resize.Bicubic(c.width, c.height, filter_param_a=1, filter_param_b=0).std.Maximum()
    return c.std.MaskedMerge(d, msk)

u = fixchroma(u, y.std.Prewitt(), get_y(src), u, v)
v = fixchroma(v, y.std.Prewitt(), get_y(src), u, v)

src = join([y, u, v])

src = depth(src, 10)

src.set_output()
