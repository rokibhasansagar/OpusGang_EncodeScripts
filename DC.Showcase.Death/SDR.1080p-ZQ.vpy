from functools import partial

import lvsfunc
import vapoursynth as vs
import vsdysfunctional
import vsmask
import vsutil
from adptvgrnMod import adptvgrnMod
from awsmfunc import ScreenGen, autogma, fixlvls
from EoEfunc.denoise import CMDegrain
from jvsfunc import ccdmod
from muvsfunc import GradFun3
from vsaa import upscaled_sraa
from vsdehalo import fine_dehalo
from vsdpir_ncnn import dpir
from vsrgtools import contrasharpening, gauss_blur

core = vs.core

F3KDB = (3, 5, 10)

INTRO = (56, 494)
FILM = (1128, 25844)


def grain_fun(clip: vs.VideoNode,
              strength: int = 1,
              type: int = 0
              ) -> vs.VideoNode:

    grain = core.noise.Add(
        clip, var=strength, uvar=0, seed=422, constant=True, type=type
        )

    diff = core.std.MakeDiff(clip, grain)

    blur = gauss_blur(diff, sigma=1)

    merge = core.std.MergeDiff(clip, blur)

    return merge


src = core.ffms2.Source(
    'Death.20191080p.BluRay.REMUX.AVC.DD.5.1-Hoyts.mkv'
)

b16, b32 = [vsutil.depth(src, i) for i in (16, 32)]
b16_luma = vsutil.get_y(b16)

y444 = core.fmtc.resample(src, css='444', kernel='blackman')
rgb = core.resize.Bicubic(
    y444, format=vs.RGBS, matrix_in_s='709', dither_type='error_diffusion'
    )

denoise = dpir(rgb, strength=25, task='deblock')

yuv = core.resize.Bicubic(denoise, format=vs.YUV444P16, matrix_s='709')

deband_mask = vsutil.get_y(yuv)
deband_mask = vsmask.edge.FDoG().edgemask(deband_mask)
deband_mask = vsmask.util.expand(deband_mask, sw=2, sh=2)

deband_mask_gradfun = vsmask.util.expand(
    deband_mask, sw=6, sh=6
    ).std.BinarizeMask(50 << 8)

deband1 = core.neo_f3kdb.Deband(
    yuv, preset='nograin', range=12, blur_first=False,
    y2=F3KDB[2], cb2=F3KDB[2], cr2=F3KDB[2], sample_mode=4
    )

deband2 = GradFun3(yuv, thr=0.33, elast=1.4, elastc=1, radius=16, smode=1)

deband = core.std.MaskedMerge(deband2, deband1, deband_mask_gradfun)
deband = core.std.MaskedMerge(deband, yuv, deband_mask)

deband = lvsfunc.rfs(deband, yuv, ranges=[(4188, 4267)])

dehalo = fine_dehalo(deband)

subsample = core.fmtc.resample(dehalo, css='420', kernel='blackman', taps=3)

aliasing = upscaled_sraa(subsample)
aliasing = core.std.MaskedMerge(subsample, aliasing, deband_mask)
aliasing = lvsfunc.rfs(
    subsample, aliasing,
    ranges=[
        (4188, 4296), (11460, 11839), (12576, 12699)
        ]
    )

# x265 has no temporal filtering
temporal = CMDegrain(aliasing, tr=3, thSAD=20, freq_merge=False)
temporal = contrasharpening(temporal, b16, radius=1, planes=0)

temporal_stronger = CMDegrain(b16, tr=2, thSAD=120, freq_merge=False)
temporal_stronger = ccdmod(temporal_stronger, threshold=6)
temporal_stronger = contrasharpening(temporal_stronger, b16)

exclude = lvsfunc.rfs(
    temporal, b16, ranges=[
        (0, INTRO[0]), (INTRO[1], FILM[0]),
        (FILM[1], src.num_frames - 1)
        ]
    )

exclude = lvsfunc.rfs(
    exclude, temporal_stronger,
    ranges=[(INTRO[1], FILM[0]), (FILM[1], src.num_frames - 1)]
)

grain = adptvgrnMod(
    exclude, luma_scaling=80,
    grainer=partial(grain_fun, strength=0.2)
    )

out = core.fmtc.bitdepth(grain, bits=10, dmode=8).set_output()