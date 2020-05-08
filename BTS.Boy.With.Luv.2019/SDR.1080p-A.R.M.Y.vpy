import vapoursynth as vs
core = vs.get_core()

import fvsfunc as fvf
from adptvgrnMod import *
from rekt import rektlvls
from functools import partial
import havsfunc as hvf

src = core.ffms2.Source("/home/a/Videos/Music_Videos/BTS - Boy With Luv (feat. Halsey) (Master).mov")

b16 = fvf.Depth(src, 16)

crp = b16.std.Crop(top=152, bottom=150)

lvl = rektlvls(crp, rownum=crp.height - 1, rowval=148, prot_val=0)

fmg = crp.fb.FillBorders(bottom=1, mode="fillmargins")

fix = core.std.ShufflePlanes([lvl, fmg, fmg], [0, 1, 2], vs.YUV)

lmk = fix.std.ShufflePlanes(0, vs.GRAY).std.Binarize(9000).std.Maximum().std.Inflate().std.Inflate()
dbn = fix.f3kdb.Deband(range=30, y=200, cr=0, cb=300, grainy=0, grainc=0, output_depth=16)
grn = adptvgrnMod(dbn, luma_scaling=3, size=1.3, grainer=lambda x: x.grain.Add(var=0.5, uvar=0.5, constant=False))
mrg = core.std.MaskedMerge(grn, fix, lmk, planes=[0, 2])
snf = fvf.rfs(fix, mrg, "[4447 4529]")

def autobrdr(n, clip, left=0, top=160, right=0, bottom=160, fix="bbmod"):
    # AutoCrop doesn't work properly with YUV422 footage.
    yuv444 = core.resize.Point(clip, format=vs.YUV444P16) if clip.format.subsampling_h == 0 and clip.format.subsampling_w == 1 else clip
    crop_values = core.acrop.CropValues(yuv444, top=top, bottom=bottom, left=left, right=right,
                                        color=[0, 120, 120], color_second=[20, 150, 150])
    cropped = core.std.Crop(clip, top=crop_values.get_frame(n).props.CropTopValue, bottom=crop_values.get_frame(n).props.CropBottomValue, 
                            left=crop_values.get_frame(n).props.CropLeftValue, right=crop_values.get_frame(n).props.CropRightValue)
    if fix is "bbmod":
        import havsfunc as hvf
        fixed = hvf.bbmod(cropped, cTop=1 if crop_values.get_frame(n).props.CropTopValue > 0 else 0, cBottom=1 if crop_values.get_frame(n).props.CropBottomValue > 0 else 0, 
                          cLeft=1 if crop_values.get_frame(n).props.CropLeftValue > 0 else 0, cRight=1 if crop_values.get_frame(n).props.CropRightValue > 0 else 0, radius=5)
    elif fix is ("cf" or "ContinuityFixer"):
        fixed = core.cf.ContinuityFixer(cropped, top=[1, 1, 1] if crop_values.get_frame(n).props.CropTopValue > 0 else 0, bottom=[1, 1, 1] if crop_values.get_frame(n).props.CropBottomValue > 0 else 0, 
                                        left=[1, 1, 1] if crop_values.get_frame(n).props.CropLeftValue > 0 else 0, right=[1, 1, 1] if crop_values.get_frame(n).props.CropRightValue > 0 else 0)
    filled = core.std.AddBorders(fixed, top=crop_values.get_frame(n).props.CropTopValue, bottom=crop_values.get_frame(n).props.CropBottomValue, 
                                 left=crop_values.get_frame(n).props.CropLeftValue, right=crop_values.get_frame(n).props.CropRightValue)
    return filled

acr = core.std.FrameEval(b16, partial(autobrdr, clip=b16, fix="cf"))

bfx = b16.std.Trim(0, 229) + acr.std.Trim(230, 407) + snf.std.AddBorders(top=152, bottom=150).std.Trim(408)

fdh = hvf.FineDehalo(bfx, showmask=0, thlimi=50, thlima=100, thmi=128, thma=150, rx=2, darkstr=1.5, excl=False)
msk = bfx.std.ShufflePlanes(0, vs.GRAY).std.Crop(top=152, bottom=150).std.Binarize(10000, 65535, 0).std.Maximum().std.Inflate().std.AddBorders(top=152, bottom=150)
mdh = bfx.std.MaskedMerge(fdh, msk)

def brghtthr(n, f, clip_a, clip_b, thr):
    if f.props.PlaneStatsAverage > thr:
        return clip_a
    else:
        return clip_b

fre = core.std.FrameEval(bfx, partial(brghtthr, clip_a=mdh, clip_b=bfx, thr=0.4), prop_src=crp.std.ShufflePlanes(0, vs.GRAY).std.PlaneStats())

out = fvf.Depth(fre, 10)

out.set_output()