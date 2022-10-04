from functools import partial

import lvsfunc
import vapoursynth as vs
import vsmask
import vsrgtools
import vsutil
from adptvgrnMod import adptvgrnMod
from debandshit import f3kbilateral
from EoEfunc.denoise import CMDegrain
from jvsfunc import ccdmod
from muvsfunc import GradFun3
from vsaa import upscaled_sraa
from vsdehalo import fine_dehalo
from vsdpir_ncnn import dpir
from vsrgtools import RepairMode, contrasharpening, gauss_blur

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

disaster = core.ffms2.Source(
    'disaster.mkv'
)

b16, b32 = [vsutil.depth(src, i) for i in (16, 32)]
b16_luma = vsutil.get_y(b16)

y444 = core.fmtc.resample(b32, css='444', kernel='blackman')
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

deband = contrasharpening(deband, yuv, radius=2, planes=0)

deband = lvsfunc.rfs(deband, yuv, ranges=[(4188, 4267)])

emergency_deband = f3kbilateral(deband)
deband = lvsfunc.rfs(deband, emergency_deband, ranges=[(27321, 27331)])

dehalo = fine_dehalo(deband)

#pre_disaster = YAHR(b16)
#pre_disaster = fine_dehalo(pre_disaster)
#
#pre_disaster_y444 = core.fmtc.resample(pre_disaster, css='444', kernel='blackman')
#
#disaster = core.resize.Bicubic(pre_disaster_y444, format=vs.RGBS, matrix_in_s='709')
#disaster = dpir(disaster, strength=100, task='deblock')
#disaster = core.resize.Bicubic(disaster, format=pre_disaster_y444.format, matrix_s='709')

y444_16 = vsutil.depth(y444, 16)
disaster = y444_16[0:23704] + disaster[1:disaster.num_frames - 1] + y444_16[23835:]

disaster_mask = vsmask.edge.FDoG(
    ).edgemask(b16_luma, multi=2).std.BinarizeMask()
disaster_mask = vsutil.iterate(disaster_mask, core.std.Maximum, 2)

merge = core.std.MaskedMerge(deband, disaster, disaster_mask)

disaster_repair = fine_dehalo(merge)

dehalo = lvsfunc.rfs(dehalo, disaster_repair, ranges=[(23704, 23835)])

subsample = core.fmtc.resample(dehalo, css='420', kernel='blackman', taps=3)

aliasing = upscaled_sraa(subsample)
aliasing = core.std.MaskedMerge(subsample, aliasing, deband_mask)
aliasing = lvsfunc.rfs(
    subsample, aliasing,
    ranges=[
        (4188, 4296), (11460, 11839), (12576, 12699), (61, 90), (27344, 27387), (21305,21538),
        (22806,22952), (23959,24054), (24983,25162)
        ]
    )

# x265 has no temporal filtering
temporal = CMDegrain(aliasing, tr=3, thSAD=20, freq_merge=False)
temporal = core.cas.CAS(temporal)
temporal = vsrgtools.repair(temporal, aliasing, mode=RepairMode.MINMAX_SQUARE_REF3)

temporal_stronger = CMDegrain(b16, tr=2, thSAD=120, freq_merge=False)
temporal_stronger = ccdmod(temporal_stronger, threshold=6)
temporal_stronger = contrasharpening(temporal_stronger, b16)

exclude = lvsfunc.rfs(
    temporal, b16,
    ranges=[
        (0, INTRO[0]), (INTRO[1], FILM[0]), (FILM[1], src.num_frames - 1)
        ]
    )

exclude = lvsfunc.rfs(
    exclude, temporal_stronger,
    ranges=[
        (INTRO[1], FILM[0]), (FILM[1], src.num_frames - 1)
        ]
    )

grain = adptvgrnMod(
    exclude, luma_scaling=80,
    grainer=partial(grain_fun, strength=0.2)
    )

out = core.fmtc.bitdepth(grain, bits=10, dmode=8)

out.set_output()
