import vapoursynth as vs
from typing import Any, Dict, Optional

import vsutil
import lvsfunc

core = vs.core


def nnedi3_rpow2_placebo(clip, rfactor=2, width=None, height=None, correct_shift=True,
                 kernel="spline36", nsize=0, nns=3, qual=None, etype=None, pscrn=None,
                 opt=True, int16_prescreener=None, int16_predictor=None, exp=None, upsizer=None):
    """nnedi3_rpow2 is for enlarging images by powers of 2.
    Args:
        rfactor (int): Image enlargement factor.
            Must be a power of 2 in the range [2 to 1024].
        correct_shift (bool): If False, the shift is not corrected.
            The correction is accomplished by using the subpixel
            cropping capability of fmtc's resizers.
        width (int): If correcting the image center shift by using the
            "correct_shift" parameter, width/height allow you to set a
            new output resolution.
        kernel (string): Sets the resizer used for correcting the image
            center shift that nnedi3_rpow2 introduces. This can be any of
            fmtc kernels, such as "cubic", "spline36", etc.
            spline36 is the default one.
        nnedi3_args (mixed): For help with nnedi3 args
            refert to nnedi3 documentation.
        upsizer (string): Which implementation to use: nnedi3, znedi3 or nnedi3cl.
            If not selected the fastest available one will be chosen.
    """

    if width is None:
        width = clip.width * rfactor
    if height is None:
        height = clip.height * rfactor
    hshift = 0.0
    vshift = -0.5
    pkdnnedi = dict(
        dh=True,
        nsize=nsize,
        nns=nns,
        qual=qual,
        etype=etype,
        pscrn=pscrn,
    )
    pkdchroma = dict(filter=kernel, sy=-0.5)

    tmp = 1
    times = 0
    while tmp < rfactor:
        tmp *= 2
        times += 1

    if rfactor < 2 or rfactor > 1024:
        raise ValueError("nnedi3_rpow2: rfactor must be between 2 and 1024.")

    if tmp != rfactor:
        raise ValueError("nnedi3_rpow2: rfactor must be a power of 2.")

    if hasattr(core, "nnedi3cl") is True and (upsizer is None or upsizer == "nnedi3cl"):
        nnedi3 = core.nnedi3cl.NNEDI3CL
    elif hasattr(core, "znedi3") is True and (upsizer is None or upsizer == "znedi3"):
        nnedi3 = core.znedi3.nnedi3
        pkdnnedi.update(
            opt=opt,
            int16_prescreener=int16_prescreener,
            int16_predictor=int16_predictor,
            exp=exp,
        )
    elif hasattr(core, "nnedi3") is True and (upsizer is None or upsizer == "nnedi3"):
        nnedi3 = core.nnedi3.nnedi3
        pkdnnedi.update(
            opt=opt,
            int16_prescreener=int16_prescreener,
            int16_predictor=int16_predictor,
            exp=exp,
        )
    else:
        if upsizer is not None:
            print(f"nnedi3_rpow2: You chose \"{upsizer}\" but it cannot be found.")
        raise RuntimeError("nnedi3_rpow2: nnedi3/znedi3/nnedi3cl plugin is required.")

    if correct_shift or clip.format.subsampling_h:
        if hasattr(core, "fmtc") is not True:
            raise RuntimeError("nnedi3_rpow2: fmtconv plugin is required.")

    last = clip

    for i in range(times):
        field = 1 if i == 0 else 0
        last = nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)
        if last.format.subsampling_w:
            # Apparently always using field=1 for the horizontal pass somehow
            # keeps luma/chroma alignment.
            field = 1
            hshift = hshift * 2 - 0.5
        else:
            hshift = -0.5
        last = nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)

    if clip.format.subsampling_h:
        last = core.placebo.Resample(last, width=last.width, height=last.height, **pkdchroma)

    if correct_shift is True:
        last = core.placebo.Resample(last, width=width, height=height, filter=kernel, sx=vshift, sy=vshift)
        
    if last.format.id != clip.format.id:
        last = core.fmtc.bitdepth(last, csp=clip.format.id)
        
    return last
   

def rescale(clip: vs.VideoNode, width: int = 1280, height: int = 720, 
            kernel=lvsfunc.kernels.Bilinear(), threshold=0.05, show_mask=False) -> vs.VideoNode:
    descale = lvsfunc.scale.descale(clip, width=width, height=height, kernel=kernel, upscaler=None, mask=None)
    
    upscale = nnedi3_rpow2_placebo(descale, rfactor=2, width=1920, height=1080, kernel="spline64", upsizer='nnedi3cl')
    return core.std.ShufflePlanes([upscale, clip], [0,1,2], vs.YUV)
             

def SM3D(clip: vs.VideoNode, sigma: float = 3) -> vs.VideoNode:
    from vsutil import get_y
    from havsfunc import SMDegrain
    from lvsfunc.util import quick_resample
    
    luma = get_y(clip)
    ref = quick_resample(luma, lambda d: SMDegrain(d, tr=1, thSAD=115))
    
    denoise = core.bm3dcuda_rtc.BM3D(luma, ref, sigma=[sigma, 0, 0], fast=False, extractor_exp=8, transform_1d_s='DCT', 
                                     transform_2d_s='DCT', block_step=5, bm_range=9, radius=2, ps_num=2, ps_range=5)
    denoise = denoise.bm3d.VAggregate(radius=2, sample=1)
    return core.std.ShufflePlanes([denoise, clip], [0,1,2], vs.YUV)


def retinex(clip: vs.VideoNode, tsigma: float = 1.5, rsigma: list[float] = [50, 200, 350], opencl: bool = False,
            msrcp_dict: Optional[Dict[str, Any]] = None, tcanny_dict: Optional[Dict[str, Any]] = None) -> vs.VideoNode:
    from lvsfunc.util import quick_resample
    from vsutil import depth
    
    tcanny = core.tcanny.TCannyCL if opencl else core.tcanny.TCanny
    
    msrcp_args: Dict[str, Any] = dict(upper_thr=0.005, fulls=True)
    if msrcp_dict is not None:
        msrcp_args |= msrcp_dict

    tcanny_args: Dict[str, Any] = dict(mode=1)
    if tcanny_dict is not None:
        tcanny_args |= tcanny_dict
        
    if clip.format.bits_per_sample == 32: 
        max_value = 1
    else: 
        max_value = vsutil.scale_value(1, 32, clip.format.bits_per_sample, scale_offsets=True, range=1)
    
    if clip.format.num_planes > 1:
        clip = vsutil.get_y(clip)
    
    ret = quick_resample(clip, lambda x: core.retinex.MSRCP(x, sigma=rsigma, **msrcp_args))
    tcanny = tcanny(ret, sigma=tsigma, **tcanny_args).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
    
    return depth(core.std.Expr([clip, tcanny], f'x y + {max_value} min'), clip.format.bits_per_sample, dither_type='none')


def sraa(clip: vs.VideoNode, rfactor: list[float] = [1.5, 1.5, 1.5], planes: list[int] = [0,1,2]) -> vs.VideoNode:
    from typing import Callable
    from lvsfunc.aa import upscaled_sraa
    from vsutil import get_y, split, join, get_w
    from functools import partial

    
    if isinstance(rfactor, float): rfactor=[rfactor, rfactor, rfactor]
    
    def nnedi3(opencl: bool = True, **override: Any) -> Callable[[vs.VideoNode], vs.VideoNode]:
        
        nnedi3_args: Dict[str, Any] = dict(field=0, dh=True, nsize=3, nns=3, qual=1)
        nnedi3_args.update(override)
    
        def _nnedi3(clip: vs.VideoNode) -> vs.VideoNode:
            return clip.nnedi3cl.NNEDI3CL(**nnedi3_args) if opencl \
                else clip.nnedi3.nnedi3(**nnedi3_args)

        return _nnedi3
    
    def _nnedi3_supersample(clip: vs.VideoNode, width: int, height: int, opencl: bool = True) -> vs.VideoNode:
        
        nnargs: Dict[str, Any] = dict(field=0, dh=True, nsize=0, nns=4, qual=2)
        _nnedi3 = nnedi3(opencl=opencl, **nnargs)
        up_y = _nnedi3(get_y(clip))
        up_y = up_y.resize.Spline36(height=height, src_top=0.5).std.Transpose()
        up_y = _nnedi3(up_y)
        up_y = up_y.resize.Spline36(height=width, src_top=0.5)
        return up_y
    
    
    def eedi3(opencl: bool = True, **override: Any) -> Callable[[vs.VideoNode], vs.VideoNode]:

        eedi3_args: Dict[str, Any] = dict(field=0, dh=True, alpha=0.25, beta=0.5, gamma=40, nrad=2, mdis=20)
        eedi3_args.update(override)
    
        def _eedi3(clip: vs.VideoNode) -> vs.VideoNode:
            return clip.eedi3m.EEDI3CL(**eedi3_args) if opencl \
                else clip.eedi3m.EEDI3(**eedi3_args)
                
        return _eedi3
    
    def _eedi3_singlerate(clip: vs.VideoNode, opencl: bool = False) -> vs.VideoNode:
        eeargs: Dict[str, Any] = dict(field=0, dh=False, alpha=0.2, beta=0.6, gamma=40, nrad=2, mdis=20)
        nnargs: Dict[str, Any] = dict(field=0, dh=False, nsize=0, nns=4, qual=2)
        y = get_y(clip)
        return eedi3(sclip=nnedi3(**nnargs)(y), **eeargs, opencl=opencl)(y)
    
    planar = split(clip)

    if 0 in planes: planar[0] = upscaled_sraa(planar[0], rfactor=rfactor[0], supersampler=partial(_nnedi3_supersample, opencl=True), aafun=partial(_eedi3_singlerate, opencl=True))
    if 1 in planes: planar[1] = upscaled_sraa(planar[1], rfactor=rfactor[1], supersampler=partial(_nnedi3_supersample, opencl=False), aafun=partial(_eedi3_singlerate, opencl=False))
    if 2 in planes: planar[2] = upscaled_sraa(planar[2], rfactor=rfactor[2], supersampler=partial(_nnedi3_supersample, opencl=False), aafun=partial(_eedi3_singlerate, opencl=False))
    
    return join(planar)