from functools import partial

import awsmfunc
import EoEfunc
import havsfunc
import jvsfunc
import vapoursynth as vs
import vardefunc
import vsdehalo
import vsdysfunctional
import vsmask
import vsutil
from vsdenoise import knl_means_cl

core = vs.core


def prep(clip: vs.VideoNode) -> vs.VideoNode:
    source = core.lsmas.LWLibavSource(clip)
    crop = core.std.Crop(source, top=128, bottom=128)
    return vsutil.depth(crop, 16)


def mvMask(clip: vs.VideoNode) -> vs.VideoNode:
    # interleave input
    mvSuper = core.mv.Super(clip, sharp=1, pel=2)

    mvAnalyze1 = core.mv.Analyse(mvSuper, isb=True, blksize=8, dct=5, searchparam=5)
    mvAnalyze2 = core.mv.Analyse(mvSuper, isb=False, blksize=8, dct=5, searchparam=5)

    mvMask1 = core.mv.Mask(clip, vectors=mvAnalyze1, kind=0, gamma=2.0, ml=1)[1::2]
    mvMask2 = core.mv.Mask(clip, vectors=mvAnalyze2, kind=0, gamma=2.0, ml=1)[1::2]
    mvMask1, mvMask2 = [vsutil.get_y(x) for x in (mvMask1, mvMask2)]
    mvmask = havsfunc.Overlay(mvMask1, mvMask2, mode='overlay')
    return mvmask


def _fun(n, f, clip1, clip2, thr: int = 36) -> None:
    return clip1 if f.props['psnr_hvs'] > thr else clip2


sources = [
    "00002.m2ts",
    '00136.m2ts'
]

USA, FRA = [prep(x) for x in sources]
USA, FRA = USA[814:], FRA[:FRA.num_frames - 19]

clean = [knl_means_cl(vsutil.get_y(x), strength=4) for x in (USA, FRA)]
weve = core.std.Interleave([clean[0], clean[1]])
weve = havsfunc.DitherLumaRebuild(weve, s0=1.0, c=0.0625)

mMask = core.motionmask.MotionMask(weve, th1=1, th2=2)[1::2]
mMask = vsmask.util.inpand(mMask, sw=2, sh=2)
mMask = vsmask.util.expand(mMask, sw=4, sh=4)

# shits HORRENDOUS
deblock = havsfunc.Deblock_QED(FRA, quant1=32)
merge = core.std.MaskedMerge(USA, deblock, mMask)

# psnr_hvs does not work at 16bpp
# https://github.com/Netflix/vmaf/pull/1050
score = core.vmaf.Metric(
    reference=vsutil.depth(USA, 8, dither_type='none'),
    distorted=vsutil.depth(FRA, 8, dither_type='none'),
    feature=1
    )
# there are a few scenes where the FRA release is
# so bad that the mask is picking up on the difference
# so lets exclude those scenes automatically
merge = core.std.FrameEval(
    merge, partial(_fun, clip1=merge, clip2=USA, thr=36), score
    )
# borders are all kinds of fucked up and I am _not_ going to fix them manually
dirty = awsmfunc.FillBorders(merge, left=1, bottom=1, planes=[0, 1, 2])
dirty = awsmfunc.bbmod(dirty, left=2, bottom=2, blur=999, planes=[0, 1, 2])
dirty = awsmfunc.bbmod(dirty, left=12, blur=999, planes=[1, 2])

# source has a bunch of halos from whatever shitty upscale they've done
# usually I would want to do this after resampling with a sharp kenrel
# however the input would likely not be accurate enough after resampling
# ssim_downsample generally doesn't ring much anyway
chroma = vsdehalo.fine_dehalo(dirty, brightstr=4/3, darkstr=2/3)

# sinc-based kernels are either too blurry or too ring-y so lets use something better
resize = vsdysfunctional.resample.ssimBetter(
    chroma, preset=None, width=1280,
    height=550, smooth=0.4, chroma=True,
    postfilter=partial(knl_means_cl, strength=4)
    )

chroma = EoEfunc.denoise.CMDegrain(
    resize, tr=4, thSADC=150, freq_merge=False, planes=[1, 2], contrasharp=True
    )

with vardefunc.YUVPlanes(chroma) as split:
    #prefilter = jvsfunc.sbr(split.Y, r=3)
    prefilter = clean[0].resize.Lanczos(split.Y.width, split.Y.height)
    binarize = core.std.Binarize(prefilter, threshold=150 << 8, planes=0)
    edge = vsmask.edge.FDoG().edgemask(prefilter)
    edge = vsmask.util.expand(edge, sw=2, sh=2)
    mask = core.std.Expr([binarize, edge], expr=["x y -"])

    denoise = vsdysfunctional.denoise.CoolDegrain(
        split.Y, tr=3, thSAD=150, planes=0
        )
    split.Y = core.std.MaskedMerge(
        split.Y, denoise, mask
    )

merged = split.clip

vsdysfunctional.misc.output(
    [USA, FRA, score, merged], operation=lambda x: x.resize.Spline36(USA.width, USA.height), debug=True
)
