import awsmfunc
import debandshit
import EoEfunc
import lvsfunc
import vapoursynth as vs
import vardefunc
import vsdysfunctional
import vsffdnet
import vsmask
import vsutil
import rekt
from havsfunc import HQDeringmod
from lvsfunc import chickendream
from lvsfunc.types import Matrix
from muvsfunc import MaskedLimitFilter
from vardefunc import AddGrain, Graigasm
from vsdenoise import Profile, BM3DCuda
from adptvgrnMod import adptvgrnMod

core = vs.core

cuts = [
    (2699, 2718),
    (85421, 85441),
    (41703, 41720),
    (151875, 151921),
    (151433, 151457),
    (42153, 42181),
    (140551, 140579),
    (145277, 145289),
    (54996, 55017),
    (197458, 197486),
    (84724, 84739),
    (204449, 204461),
    (201480, 201493),
    (203576, 203590),
    (171240, 171301),
    (170735, 170747),
    (170834, 170859)
    ]


def nuke_grain(
        clip: vs.VideoNode, tr: int = 3,
        thSAD: int = 500, sigma: int = 20
        ) -> vs.VideoNode:

    bm3d_pre = EoEfunc.denoise.CMDegrain(
        clip, tr=tr, thSAD=thSAD, freq_merge=False, truemotion=True)

    bm3d = BM3DCuda(clip, sigma=sigma, ref=bm3d_pre, profile=Profile.HIGH)
    bm3d.basic_args = dict(chroma=True, fast=False)
    bm3d.final_args = dict(chroma=True, fast=False)
    bm3d = bm3d.clip

    return bm3d


def ml_magic(clip: vs.VideoNode, strength: float = 5) -> vs.VideoNode:

    y444 = vardefunc.scale.to_444(
        clip, width=1920, height=1080, join_planes=True
        )
    rgb = y444.resize.Point(format=vs.RGBS)

    ffd = vsffdnet.FFDNet(rgb, strength=strength)

    yuv = core.resize.Point(
        ffd, format=vs.YUV444P16, matrix_s='709'
    )

    repair = EoEfunc.denoise.CMDegrain(
        yuv, tr=2, thSAD=30, contrasharp=y444,
        refine=4, truemotion=True, freq_merge=False
        ).fmtc.resample(css="420", kernel="blackmanminlobe")

    return repair


def dnrfun(clip: vs.VideoNode, sigma: list[float] | float = 4) -> vs.VideoNode:

    clip = vardefunc.scale.to_444(
        clip, width=1920, height=1080, join_planes=True
        )

    bm3d = BM3DCuda(clip, sigma=sigma, profile=Profile.HIGH)
    bm3d.basic_args = dict(chroma=True, fast=False)
    bm3d.final_args = dict(chroma=True, fast=False)
    bm3d = bm3d.clip

    repair = EoEfunc.denoise.CMDegrain(
        bm3d, tr=2, thSAD=60, contrasharp=clip,
        refine=4, truemotion=True, freq_merge=False
        ).fmtc.resample(css="420", kernel="blackmanminlobe")

    return repair


def average_diff(clip: vs.VideoNode, diff: vs.VideoNode) -> vs.VideoNode:
    store = core.std.MakeDiff(clip, diff)
    average = core.std.AverageFrames(store, weights=[1] * 3)
    regrain = core.std.MergeDiff(clip, average)
    return regrain


src = core.lsmas.LWLibavSource('.mkv')
src = core.std.SetFrameProps(src, _Matrix=1, _ColorRange=1)
src = vsutil.depth(src, 16)

sdr = core.lsmas.LWLibavSource(
    '.mkv'
).std.AddBorders(top=140, bottom=140)
sdr = vsutil.depth(sdr, 16)
sdr = sdr[(24 * 14) + 2:]

dnr_bm3d = dnrfun(src, sigma=3)
dnr_ffd = ml_magic(src, strength=3)
ffd = lvsfunc.rfs(dnr_bm3d, dnr_ffd, ranges=[
    (3119, 4205), (5296, 5304), (9214, 9723),
    (13063, 13200), (14317, 14546), (14667, 14684), (15672, 16103),
    (17145, 17542), (17700, 17807), (17620, 17699), (17938, 18009), (18850, 19041)
    ])

truck_scene = nuke_grain(
    src, tr=4, thSAD=500, sigma=20)
truck_scene = EoEfunc.denoise.CMDegrain(
    truck_scene, tr=3, thSAD=100, freq_merge=False, contrasharp=src
).f3kdb.Deband(preset='nograin')

truck_scene = core.std.MaskedMerge(
    ffd, truck_scene, mask=vsutil.get_y(src).std.Binarize(60 << 8)
)
truck_scene = adptvgrnMod(
    truck_scene, strength=[1, 0.35], size=1.5,
    static=False, temporal_average=30, luma_scaling=0)

ffd = lvsfunc.rfs(
    ffd, truck_scene, ranges=[(4775, 4936), (5140, 5324)]
)

deband = debandshit.dumb3kdb(ffd, radius=20, threshold=[35, 50, 50])
deband = vsdysfunctional.util.csharp(deband, ffd)
deband = MaskedLimitFilter(deband, ffd)
process = lvsfunc.rfs(
    ffd, deband, ranges=[
        (948, 1034), (1225, 1377), (6265, 6338),
        (9214, 9727), (12141, 12357), (13333, 13392)
    ]
)

bm3d = nuke_grain(src)

regrain = chickendream(
    bm3d, dev=0.6, rad=0.04, draft=True, matrix=Matrix.BT709
    )
regrain = average_diff(bm3d, regrain)

exclude = lvsfunc.rfs(
    process, src, ranges=[
            (2041, 2198), (2235, 2422),
            (2459, 2866), (8622, 8622),
            (9751, 12131)
        ]
    )

exclude = lvsfunc.rfs(
    exclude, regrain, ranges=[(9751, 12131)]
)

aa_luma = vsutil.get_y(ffd)
aa_mask = vsmask.edge.FDoG().edgemask(aa_luma)
aa_mask = vsmask.util.expand(aa_mask, sw=3, sh=3).std.Binarize()

aa = vsutil.iterate(exclude, lvsfunc.sraa, 2)
aa = core.std.MaskedMerge(exclude, aa, aa_mask)
aa = lvsfunc.rfs(
    exclude, aa,
    ranges=[
        (3950, 3994), (6102, 6180),
        (6339, 6414), (6416, 6573),
        (7414, 7461), (7846, 7917),
        (8569, 8724), (9315, 9447),
        (13563, 13602), (13644, 13673),
        (13888, 13916), (14257, 14294),
        (14744, 14829), (15768, 15851),
        (16472, 16602),
    ])

ringing_text = rekt.rekt_fast(
    src, fun=lambda x: HQDeringmod(x, mrad=8, sigma=256, thr=24, darkthr=24),
    left=600, right=600, top=720, bottom=278
    )

ringing = core.bilateral.Bilateral(aa)
ringmask = lvsfunc.mask.halo_mask(ringing)
ringing = core.std.MaskedMerge(
    aa, ringing, mask=ringmask
    )

ringing = lvsfunc.rfs(
    ringing, ringing_text,
    ranges=[(6763, 6831)]
)

grain = Graigasm(
    thrs=[x << 8 for x in (16, 90, 200)],
    strengths=[(3, .5), (1.5, .5), (0, 0)],
    sizes=(1.5, 1.5, 1.5),
    sharps=(50, 50, 50),
    grainers=[
        AddGrain(seed=422, constant=True),
        AddGrain(seed=422, constant=True),
        AddGrain(seed=422, constant=True)
    ],
    overflows=1.0
    ).graining(ringing, show_masks=False)

restore = core.std.Crop(src, top=140, bottom=140)
restore = awsmfunc.bbmod(
    restore, top=6, bottom=8, blur=999, planes=[0, 1, 2]
    )
restore = EoEfunc.denoise.CMDegrain(
    restore, tr=2, thSAD=100, freq_merge=False
    )
restore_noise = lvsfunc.chickendream(
    restore, rad=0.015, dev=0.1, draft=True
    )
restore = core.std.Merge(restore, restore_noise, 0.7)
restore = awsmfunc.AddBordersMod(restore, top=140, bottom=140)

# can you tell this was an afterthought?
# this should be possible with a loop or two,
# I just don't care at this point

cuts = grain[0:72] + sdr[cuts[0][0]:cuts[0][1]] + grain[91:103] \
    + sdr[cuts[1][0]:cuts[1][1]] + sdr[cuts[2][0]:cuts[2][1]] \
    + sdr[cuts[3][0]:cuts[3][1]] + sdr[cuts[4][0]:cuts[4][1]] \
    + grain[210:278] + sdr[cuts[6][0]:cuts[6][1]] + grain[306:317] \
    + sdr[cuts[7][0]:cuts[7][1]] + sdr[cuts[8][0]:cuts[8][1]] \
    + sdr[cuts[9][0]:cuts[9][1]] + sdr[cuts[10][0]:cuts[10][1]] \
    + restore[393:454] + sdr[cuts[11][0]:cuts[11][1]] + restore[466:478] \
    + sdr[cuts[12][0]:cuts[12][1]] + sdr[cuts[13][0]:cuts[13][1]] \
    + restore[505:517] + grain[517:564] + restore[564:601] \
    + sdr[cuts[15][0]:cuts[15][1]] + sdr[cuts[16][0]:cuts[16][1]] + grain[638:]

out = core.fmtc.bitdepth(cuts, bits=8, dmode=9)
out.set_output()


