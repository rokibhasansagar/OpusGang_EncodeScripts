import vapoursynth as vs

import awsmfunc
import lvsfunc
import vsutil
import dysfunctional
import zzfunc

from adptvgrnMod import *
from zzfunc import minfilter
from fvsfunc import GradFun3
from kagefunc import retinex_edgemask
from havsfunc import FineDehalo
from vsutil import scale_value

core = vs.core

def bbmod1(clip, top=0, bottom=0, left=0, right=0, thresh=None, blur=20, planes=None, y=None, u=None,
          v=None, scale_thresh=None, cpass2=False, csize=2, scale_offsets=True, cTop=None, cBottom=None,
          cLeft=None, cRight=None):
    """
    Narkyy's bbmod helper for a significant speedup from cropping unnecessary pixels before processing.
    :param clip: Clip to be processed.
    :param top: Top rows to be processed.
    :param bottom: Bottom rows to be processed.
    :param left: Left columns to be processed.
    :param right: Right columns to be processed.
    :param thresh: Largest change allowed. Scale of 0-128, default is 128 (assuming 8-bit).
                   Specify a list for [luma, chroma] or [y, u, v].
    :param blur: Processing strength, lower values are more aggressive. Default is 20, not 999 like the old bbmod.
                 Specify a list for [luma, chroma] or [y, u, v].
    :param planes: Planes to process. Overwrites y, u, v. Defaults to all planes.
    :param y: Boolean whether luma plane is processed. Default is True.
    :param u: Boolean whether first chroma plane is processed. Default is True.
    :param v: Boolean whether second chroma plane is processed. Default is True.
    :param scale_thresh: Boolean whether thresh value is scaled from 8-bit to source bit depth.
                         If thresh <= 128, this defaults to True, else False.
    :param cpass2: Second, significantly stronger, chroma pass. If enabled, default for chroma blur is blur * 2 and
                   chroma thresh is thresh / 10.
    :param csize: Size to be cropped to. This might help maintain details at the cost of processing speed.
    :param scale_offsets: Whether scaling should take offsets into account in vsutil.scale_value.
                          If you don't know what this means, don't change it.  Thresh never uses scale_offsets.
    :param cTop: Legacy top.
    :param cBottom: Legacy bottom.
    :param cLeft: Legacy left.
    :param cRight: Legacy right.
    :return: Clip with color offsets fixed.
    """

    if clip.format.color_family != vs.YUV and clip.format.color_family != vs.GRAY:
        raise ValueError("bbmod: only YUV and GRAY clips are supported")

    if cTop is not None:
        top = cTop

    if cBottom is not None:
        bottom = cBottom

    if cLeft is not None:
        left = cLeft

    if cRight is not None:
        right = cRight

    if planes is not None:
        if isinstance(planes, int):
            planes = [planes]
    
        if 0 in planes:
            y = True
        else:
            y = False
    
        if 1 in planes:
            u = True
        else:
            u = False
    
        if 2 in planes:
            v = True
        else:
            v = False
    elif clip.format.color_family == vs.YUV:
        if y is None:
            y = True
        if u is None:
            u = True
        if v is None:
            v = True
    else:
        if y is None and not u and not v:
            y = True
            u = False
            v = False
        elif y == True:
            u = False
            v = False
        else:
            y = False
            u = True
            v = False

    depth = clip.format.bits_per_sample
    if thresh is None:
        thresh = scale_value(128, 8, depth, scale_offsets=scale_offsets)

    if scale_thresh is None:
        if thresh < 1:
            scale_thresh = False
        elif thresh < 129:
            scale_thresh = True
        else:
            scale_thresh = False

    filtered = clip

    c_left = max(left * csize, 4)
    c_right = max(right * csize, 4)
    c_top = max(top * csize, 4)
    c_bottom = max(bottom * csize, 4)

    f_width, f_height = filtered.width, filtered.height

    if left > 0 and right > 0:
        l = filtered.std.Crop(left=0, right=f_width - c_left, top=0, bottom=0)
        m = filtered.std.Crop(left=c_left, right=c_right, top=0, bottom=0)
        r = filtered.std.Crop(left=f_width - c_right, right=0, top=0, bottom=0)

        l = bbmoda(l, cTop=0, cBottom=0, cLeft=left, cRight=0, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)
        r = bbmoda(r, cTop=0, cBottom=0, cLeft=0, cRight=right, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)

        filtered = core.std.StackHorizontal(clips=[l, m, r])

    if left > 0 and right == 0:
        l = filtered.std.Crop(left=0, right=f_width - c_left, top=0, bottom=0)
        m = filtered.std.Crop(left=c_left, right=0, top=0, bottom=0)

        l = bbmoda(l, cTop=0, cBottom=0, cLeft=left, cRight=0, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)

        filtered = core.std.StackHorizontal(clips=[l, m])

    if left == 0 and right > 0:
        r = filtered.std.Crop(left=f_width - c_right, right=0, top=0, bottom=0)
        m = filtered.std.Crop(left=0, right=c_right, top=0, bottom=0)

        r = bbmoda(r, cTop=0, cBottom=0, cLeft=0, cRight=right, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)

        filtered = core.std.StackHorizontal(clips=[m, r])

    if top > 0 and bottom > 0:
        t = filtered.std.Crop(left=0, right=0, top=0, bottom=f_height - c_top)
        m = filtered.std.Crop(left=0, right=0, top=c_top, bottom=c_bottom)
        b = filtered.std.Crop(left=0, right=0, top=f_height - c_bottom, bottom=0)

        t = bbmoda(t, cTop=top, cBottom=0, cLeft=0, cRight=0, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)
        b = bbmoda(b, cTop=0, cBottom=bottom, cLeft=0, cRight=0, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)

        filtered = core.std.StackVertical(clips=[t, m, b])

    if top > 0 and bottom == 0:
        t = filtered.std.Crop(left=0, right=0, top=0, bottom=f_height - c_top)
        m = filtered.std.Crop(left=0, right=0, top=c_top, bottom=0)

        t = bbmoda(t, cTop=top, cBottom=0, cLeft=0, cRight=0, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)

        filtered = core.std.StackVertical(clips=[t, m])

    if top == 0 and bottom > 0:
        b = filtered.std.Crop(left=0, right=0, top=f_height - c_bottom, bottom=0)
        m = filtered.std.Crop(left=0, right=0, top=0, bottom=c_bottom)

        b = bbmoda(b, cTop=0, cBottom=bottom, cLeft=0, cRight=0, thresh=thresh, blur=blur, y=y, u=u, v=v,
                   scale_thresh=scale_thresh, cpass2=cpass2, csize=csize, scale_offsets=scale_offsets)

        filtered = core.std.StackVertical(clips=[m, b])

    return filtered

def bbmoda(c, cTop=0, cBottom=0, cLeft=0, cRight=0, thresh=128, blur=999, y=True, u=True, v=True, scale_thresh=True,
           cpass2=False, csize=2, scale_offsets=True):
    """
    From sgvsfunc. I'm not updating the doc strings, here, read bbmod instead.
    bbmod, port from Avisynth's function, a mod of BalanceBorders
      The function changes the extreme pixels of the clip, to fix or attenuate dirty borders
      Any bit depth
      Inspired from BalanceBorders from https://github.com/WolframRhodium/muvsfunc/ and https://github.com/fdar0536/Vapoursynth-BalanceBorders/
    > Usage: bbmod(c, cTop, cBottom, cLeft, cRight, thresh, blur)
      * c: Input clip. The image area "in the middle" does not change during processing.
           The clip can be any format, which differs from Avisynth's equivalent.
      * cTop, cBottom, cLeft, cRight (int, 0-inf): The number of variable pixels on each side.
      * thresh (int, 0~128, default 128): Threshold of acceptable changes for local color matching in 8 bit scale.
        Recommended: 0~16 or 128
      * blur (int, 1~inf, default 999): Degree of blur for local color matching.
        Smaller values give a more accurate color match, larger values give a more accurate picture transfer.
        Recommended: 1~20 or 999
      Notes:
        1) At default values ​​of thresh = 128 blur = 999:
           You will get a series of pixels that have been changed only by selecting the color for each row in its entirety, without local selection;
           The colors of neighboring pixels may be very different in some places, but there will be no change in the nature of the picture.
           With thresh = 128 and blur = 1 you get almost the same rows of pixels, i.e. The colors between them will coincide completely, but the original pattern will be lost.
        2) Beware of using a large number of pixels to change in combination with a high level of "thresh",
           and a small "blur" that can lead to unwanted artifacts "in a clean place".
           For each function call, try to set as few pixels as possible to change and as low a threshold as possible "thresh" (when using blur 0..16).
    """
    funcName = "bbmoda"

    if not isinstance(c, vs.VideoNode):
        raise TypeError(funcName + ': \"c\" must be a clip!')

    if isinstance(thresh, int) or isinstance(thresh, float):
        # thresh needs to be lower for chroma for cpass2
        if cpass2:
            thresh = [thresh] + 2 * [round(thresh / 10)]
        else:
            thresh = 3 * [thresh]
    elif len(thresh) == 2:
        thresh.append(thresh[1])

    if scale_thresh:
        thresh[0] = scale_value(thresh[0], 8, c.format.bits_per_sample, scale_offsets=False)
        i = 1
        for t in thresh[1:]:
            thresh[i] = scale_value(thresh[i], 8, c.format.bits_per_sample, scale_offsets=False, chroma=False)
            i += 1

    if isinstance(blur, int):
        # blur should also be higher
        if cpass2:
            blur = [blur] + 2 * [blur * 2]
        else:
            blur = 3 * [blur]
    elif len(blur) == 2:
        blur.append(blur[1])

    for _ in blur:
        if _ <= 0:
            raise ValueError(funcName + ': \'blur\' has an incorrect value! (0 ~ inf]')
    for _ in thresh:
        if _ <= 0 and c.format.sample_type == vs.INTEGER:
            raise ValueError(funcName + ': \'thresh\' has an incorrect value! (0 ~ inf]')

    def btb(c, cTop, thresh, blur):

        cWidth = c.width
        cHeight = c.height
        sw, sh = c.format.subsampling_w + 1, c.format.subsampling_h + 1
        cTop = min(cTop, cHeight - 1)
        blurWidth = [max(8, math.floor(cWidth / blur[0])), max(8, math.floor(cWidth / blur[1])),
                     max(8, math.floor(cWidth / blur[2]))]
        scale128 = str(scale_value(128, 8, c.format.bits_per_sample, scale_offsets=scale_offsets, chroma=True))
        uvexpr_ = "z y - x +"
        uvexpr = []
        for t in [1, 2]:
            uvexpr.append(uvexpr_ + " x - " +str(thresh[t])+ " > x " +str(thresh[t])+ " + " + uvexpr_ + " x - -" +str(thresh[t])+ " < x " +str(thresh[t])+ " - " + uvexpr_ + " ? ?")
        if c.format.sample_type == vs.INTEGER:
            exprchroma = f"x {scale128} - abs 2 *"
            expruv = f"z y / 8 min 0.4 max x {scale128} - * {scale128} + x - {scale128} +"
        else:
            exprchroma = f"x abs 2 *"
            expruv = "z .5 + y .5 + / 8 min .4 max x .5 + * x - .5 +"
        scale16 = str(scale_value(16, 8, c.format.bits_per_sample, scale_offsets=scale_offsets))
        yexpr = "z " + scale16 + " - y " + scale16 + " - / 8 min 0.4 max x " + scale16 + " - * " + scale16 + " +"
        yexpr = f"{yexpr} x - {thresh[0]} > x {thresh[0]} + {yexpr} x - -{thresh[0]} < x {thresh[0]} - {yexpr} ? ?"

        if y and u and v and blur[0] == blur[1] == blur[2] and thresh[0] == thresh[1] == thresh[2] and sw == sh == 1:
            c2 = core.resize.Point(c, cWidth * csize, cHeight * csize)
            last = core.std.CropAbs(c2, cWidth * csize, csize, 0, cTop * csize)
            last = core.resize.Point(last, cWidth * csize, cTop * csize)
            exprchroma = ["", exprchroma]
            if cpass2:
                referenceBlurChroma = last.std.Expr(exprchroma).fmtc.resample(blurWidth[0] * csize, cTop * csize,
                                                                               a1=20, 
                                                                               taps=6, kernel='gaussian', fh=.5, fv=.5).fmtc.resample(
                    cWidth * csize,
                    cTop * csize,
                    a1=20,
                    taps=6, kernel='gaussian', fh=.5, fv=.5)
            referenceBlur = core.fmtc.resample(last, blurWidth[0] * csize, cTop * csize, a1=20,
                                                taps=6, kernel='gaussian', fh=.5, fv=.5).fmtc.resample(cWidth * csize, cTop * csize, a1=20,
                                                                                 taps=6, kernel='gaussian', fh=.5, fv=.5)

            original = core.std.CropAbs(c2, cWidth * csize, cTop * csize, 0, 0)

            last = core.fmtc.resample(original, blurWidth[0] * csize, cTop * csize, a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)

            originalBlur = last.fmtc.resample(cWidth * csize, cTop * csize, a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)

            if cpass2:
                originalBlurChroma = last.std.Expr(exprchroma).fmtc.resample(blurWidth[0] * csize, cTop * csize, a1=20,
                                                                              taps=6, kernel='gaussian', fh=.5, fv=.5)
                originalBlurChroma = originalBlurChroma.fmtc.resample(cWidth * csize, cTop * csize, a1=20,
                                                                       taps=6, kernel='gaussian', fh=.5, fv=.5)
                balancedChroma = core.std.Expr(clips=[original, originalBlurChroma, referenceBlurChroma],
                                               expr=["", expruv])
                balancedLuma = core.std.Expr(clips=[balancedChroma, originalBlur, referenceBlur],
                                             expr=[yexpr, uvexpr[0], uvexpr[1]])
            else:
                balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur],
                                         expr=[yexpr, uvexpr[0], uvexpr[1]])

            return core.std.StackVertical(
                [balancedLuma, core.std.CropAbs(c2, cWidth * csize, (cHeight - cTop) * csize, 0, cTop * csize)]).resize.Point(
                cWidth, cHeight)
        else:
            if c.format.color_family == vs.YUV:
                yplane, uplane, vplane = split(c)
            elif c.format.color_family == vs.GRAY:
                yplane = c
            else:
                raise ValueError("bbmod: only YUV and GRAY clips are supported")
            if y:
                c2 = core.resize.Point(yplane, cWidth * csize, cHeight * csize)
                last = core.std.CropAbs(c2, cWidth * csize, csize, 0, cTop * csize)
                last = core.resize.Point(last, cWidth * csize, cTop * csize)
                referenceBlur = core.fmtc.resample(last, blurWidth[0] * csize, cTop * csize, a1=20,
                                                    taps=6, kernel='gaussian', fh=.5, fv=.5).fmtc.resample(cWidth * csize, cTop * csize,
                                                                                     a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)
                original = core.std.CropAbs(c2, cWidth * csize, cTop * csize, 0, 0)

                last = core.fmtc.resample(original, blurWidth[0] * csize, cTop * csize, a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)

                originalBlur = last.fmtc.resample(cWidth * csize, cTop * csize, a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)
                balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur], expr=yexpr)

                yplane = core.std.StackVertical(
                    clips=[balancedLuma, core.std.CropAbs(c2, cWidth * csize, (cHeight - cTop) * csize, 0, cTop * csize)]).resize.Point(
                    cWidth, cHeight)
                if c.format.color_family == vs.GRAY:
                    return yplane

            def btbc(c2, blurWidth, p, csize):
                c2 = core.resize.Point(c2, round(cWidth * csize / sw), round(cHeight * csize / sh))
                last = core.std.CropAbs(c2, round(cWidth * csize / sw), round(csize / sh), 0, round(cTop * csize / sh))
                last = core.resize.Point(last, round(cWidth * csize / sw), round(cTop * csize / sh))
                if cpass2:
                    referenceBlurChroma = last.std.Expr(exprchroma).fmtc.resample(round(blurWidth * csize / sw), round(cTop * csize / sh), a1=20,
                                                                                   taps=6, kernel='gaussian', fh=.5, fv=.5).fmtc.resample(
                        round(cWidth * csize / sw), round(cTop * csize / sh), a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)
                referenceBlur = core.fmtc.resample(last, round(blurWidth * csize / sw), round(cTop * csize / sh), a1=20,
                                                    taps=6, kernel='gaussian', fh=.5, fv=.5).fmtc.resample(round(cWidth * csize / sw), round(cTop * csize / sh), a1=20,
                                                                                     taps=6, kernel='gaussian', fh=.5, fv=.5)
                original = core.std.CropAbs(c2, round(cWidth * csize / sw), round(cTop * csize / sh), 0, 0)

                last = core.fmtc.resample(original, round(blurWidth * csize / sw), round(cTop * csize / sh), a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)

                originalBlur = last.fmtc.resample(round(cWidth * csize / sw), round(cTop * csize / sh), a1=20, taps=6, kernel='gaussian', fh=.5, fv=.5)

                if cpass2:
                    originalBlurChroma = last.std.Expr(exprchroma).fmtc.resample(round(blurWidth * csize / sw), round(cTop * csize / sh), a1=20,
                                                                                  taps=6, kernel='gaussian', fh=.5, fv=.5)
                    originalBlurChroma = originalBlurChroma.fmtc.resample(round(cWidth * csize / sw), round(cTop * csize / sh), a1=20,
                                                                           taps=6, kernel='gaussian', fh=.5, fv=.5)
                    balancedChroma = core.std.Expr(clips=[original, originalBlurChroma, referenceBlurChroma],
                                                   expr=expruv)
                    balancedLuma = core.std.Expr(clips=[balancedChroma, originalBlur, referenceBlur], expr=expruv)
                else:
                    balancedLuma = core.std.Expr(clips=[original, originalBlur, referenceBlur], expr=uvexpr[p - 1])

                return core.std.StackVertical(
                    [balancedLuma, c2.std.CropAbs(left=0, top=round(cTop * csize / sh), width=round(cWidth * csize / sw), height=round(cHeight * csize / sh) - round(cTop * csize / sh))]).resize.Point(
                    round(cWidth / sw), round(cHeight / sh))

            if c.format.color_family == vs.GRAY:
                return btbc(yplane, blurWidth[1], 1, csize)

            if u:
                uplane = btbc(uplane, blurWidth[1], 1, csize * max(sw, sh))
            if v:
                vplane = btbc(vplane, blurWidth[2], 2, csize * max(sw, sh))
            return core.std.ShufflePlanes([yplane, uplane, vplane], [0, 0, 0], vs.YUV)

    c = btb(c, cTop, thresh, blur).std.Transpose().std.FlipHorizontal() if cTop > 0 else core.std.Transpose(
        c).std.FlipHorizontal()
    c = btb(c, cLeft, thresh, blur).std.Transpose().std.FlipHorizontal() if cLeft > 0 else core.std.Transpose(
        c).std.FlipHorizontal()
    c = btb(c, cBottom, thresh, blur).std.Transpose().std.FlipHorizontal() if cBottom > 0 else core.std.Transpose(
        c).std.FlipHorizontal()
    c = btb(c, cRight, thresh, blur).std.Transpose().std.FlipHorizontal() if cRight > 0 else core.std.Transpose(
        c).std.FlipHorizontal()

    return c

src = core.ffms2.Source('')
src = core.std.Crop(src, top=104, bottom=104)
b16 = vsutil.depth(src, 16)

dirty_lines = [(28603, 28673), (34656, 35176), (68882, 68883), (93323, 93324), (120115, 120177), (120237, 120285), (120357, 120429), (120759, 120855), (121085, 121206), (135008, 135013), (135160, 135162), (140063, 140193), (144964, 145059), (164111, 164137), (164217, 164218), (164244, 164257), (165116, 165132), (165195, 165214), (165405, 165406), (165446, 165494), (165721, 165753), (165792, 165796), (166304, 166325), (166887, 166903), (167046, 167047), (167138, 167141)]
#awsmfunc.detect.dirtdtct(b16, left=0, right=0, top=0, bottom=0, thr=0.05, tolerance=24)

border1 = awsmfunc.bbmod(b16, top=3, bottom=3, left=3, right=3, planes=[0,1,2], blur=40, csize=6, thresh=10, scale_thresh=True)
border2 = core.cf.ContinuityFixer(b16, top=[3,0,0], bottom=[3,0,0], left=[3,0,0], right=[3,0,0], radius=[800]).std.Limiter(16 << 8, [235 << 8, 240 << 8]) # Apparently cf can create full range infomation
border2 = bbmod1(border2, bottom=6, blur=999, thresh=14, scale_thresh=True, planes=[1,2])
borders = zzfunc.minfilter(b16, border1, border2, planes=[0,1,2], strict=True) # ¿Por qué no los dos?

borders = lvsfunc.rfs(borders, awsmfunc.bbmod(b16, top=2, bottom=2, left=2, right=2, planes=[0,1,2], blur=32), ranges=dirty_lines)

mask = retinex_edgemask(b16)

denoise = dysfunctional.bm3dGPU(borders, sigma=2, profile='lc')
denoise = core.knlm.KNLMeansCL(denoise, h=.6, a=3, channels='uv')

#deband = dumb3kdb(denoise, radius=12, threshold=[40, 35, 35], grain=0)
deband = GradFun3(denoise, thr=0.4, thrc=0.35, radius=16, elast=2.5, smode=2)
deband = core.std.MaskedMerge(deband, denoise, mask)

hlo_mask = lvsfunc.mask.halo_mask(b16, rad=2, edgemask=dysfunctional.FDOG(b16, retinex=False, div=6, opencl=True))
hlo1 = core.neo_fft3d.FFT3D(b16, sigma=1, dehalo=1.1, sharpen=.4, hr=1, ht=55, bt=5, wintype=2, scutoff=.1, planes=[0])
hlo2 = FineDehalo(b16, brightstr=.8, darkstr=.7)
hlos = zzfunc.maxfilter(b16, hlo1, hlo2, planes=0, strict=False)
hlos = core.std.MaskedMerge(b16, hlos, hlo_mask)

grain = adptvgrnMod(hlos, strength=0.3, cstrength=0.1, static=True, seed=444, luma_scaling=6)
src.set_output(1)
vsutil.depth(grain, 10).set_output(2)