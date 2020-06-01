import vapoursynth as vs
core = vs.core
import awsmfunc as awf
import kagefunc as kgf
import fvsfunc as fvf
import vsTAAmbk as taa
from rekt import *
from vsutil import split, plane, depth
from adptvgrnMod import *

# this guy's movies usually have tons of aliasing on plants
def gmask(clip, r, g, b):
    clip = clip.resize.Spline36(format=vs.RGB48)
    rc, gc, bc = split(awf.DelFrameProp(clip))
    return core.std.Expr([rc, gc, bc], f"x {r} < y {g} > z {b} < and and 65535 0 ?")

src = core.ffms2.Source("Weathering With You 2019 1080p Bluray REMUX AVC DTS-HD MA 5.1 English Subbed -ZR-.mkv")

src = depth(src, 16)

y, u, v = split(src)

# literally unfixable
# [115802 115855]

# I'm too lazy for these
# [6158 6319] [19932 19955] [58056 58169] [80583 80640] [148509 148616]

# maybe later
# [93998 94093]  [99937 99966] [122225 122302] [122303 122429]

# anti-aliasing

# generic nnedi3 
ana = taa.TAAmbk(src, aatype=3, mtype=2, opencl=True)
snf = awf.rfs(src, ana, "[8910 9005] [11964 12083] [21126 21509] [21582 21773] [42346 42513] [53634 53741] [54544 54591] [56964 57023] [77024 77191] [91382 91489] [91664 91705] [91706 91747] [99163 99282] [99937 99966] [102211 102300] [102625 102816] [122573 122701] [123068 123271] [140157 140186] [142511 142606]")

# generic eedi3
ana = taa.TAAmbk(src, aatype=2, mtype=1, opencl=True)
snf = awf.rfs(snf, ana, "[12408 12467] [59046 59225] [91748 91795] [91796 91879] [94094 94177] [99991 100026] [115498 115557] [115738 115801] [145941 146168] [146913 147044]")

# eedi3 with larger sobel mask
ana = taa.TAAmbk(src, aatype=2, mtype=2, mthr=7, opencl=True)
snf = awf.rfs(snf, ana, "[38206 38349] [91532 91567] [114316 114375]")

# just a bunch of rektaa calls
ana = rektaa(src, 1126, 0, 462, 992, opencl=True)
snf = awf.rfs(snf, ana, "[1404 1511] [2378 2473]")

ana = rektaa(src, 162, 0, 1374, 318, aatype=2, opencl=True)
ana = rektaa(ana, 1700, 0, 0, 298, opencl=True)
snf = awf.rfs(snf, ana, "[2138 2185]")

ana = taa.TAAmbk(src, aatype=3, opencl=True, mtype=3)
ana = src.std.MaskedMerge(ana, gmask(src, 90 << 8, 50 << 8, 90 << 8))
snf = awf.rfs(snf, ana, "[3704 3751]")

ana = rektaa(src, 892, 160, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[5570 5641]")

# due to aliasing it looks like a part of an object isn't moving
# so just subtract a motion mask (12 fps) from an edge mask
mm2 = core.motionmask.MotionMask(y.std.SelectEvery(cycle=2, offsets=0)).std.SelectEvery(cycle=2, offsets=[0, 0, 1, 1]).std.Maximum().std.Maximum()
edm = y.std.Prewitt()
msk = core.std.Expr([mm2, edm], "y x -").std.Binarize(40000).std.Maximum().std.Maximum()
ana = rektaa(src, 908, 676, 960, 194, aatype=2, opencl=True)
ana = core.std.MaskedMerge(src, ana, msk)
snf = awf.rfs(snf, ana, "[5642 5773]")

ana = rektaa(src, 0, 0, 1368, 910, aatype=6, opencl=True, mtype=2)
ana = rektaa(ana, 742, 254, 976, 760, aatype=6, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[5846 5905]")

ana = rektaa(src, 0, 24, 722, 962, aatype=6, opencl=True, mtype=1)
ana = rektaa(ana, 1080, 118, 166, 904, aatype=6, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[6014 6109]")

ana = rektaa(src, 648, 552, 1232, 338, aatype=2, opencl=True, mtype=1)
ana = rektaa(ana, 730, 782, 1096, aatype=2, opencl=True, mtype=1)
ana = rektaa(ana, 1696, 404, 0, 550, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[8214 8321] [8598 8909]")

ana = rektaa(src, 348, 64, 832, 842, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[11064 11171]")

ana = rektaa(src, 0, 152, 116, 776, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[11388 11411]")

ana = rektaa(src, 0, 206, 1270, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[11412 11447]")

ana = rektaa(src, 1690, 720, 0, 268, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[11494 11507]")

ana = rektaa(src, 390, 92, 412, 784, aatype=2, opencl=True, mtype=3)
ana = src.std.MaskedMerge(ana, gmask(src, 100 << 8, 40 << 8, 110 << 8).std.Maximum())
snf = awf.rfs(snf, ana, "[11568 11603]")

# just removing very very bright areasa from this
ana = rektaa(src, 158, 216, 1374, 600, aatype=6, opencl=True, mtype=2)
ana = ana.std.MaskedMerge(src, y.std.Binarize(30000).std.Minimum())
snf = awf.rfs(snf, ana, "[11796 11867]")

ana = rektaa(src, 434, 298, 1390, 750, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[13830 13889]")

ana = rektaa(src, 314, 430, 1516, 502, aatype=6, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[13890 13961]")

ana = rektaa(src, 908, 488, 726, 554, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[14867 14974]")

# motion mask on 24 fps content
mm1 = core.motionmask.MotionMask(y).std.Maximum().std.Maximum()
edm = y.std.Prewitt()
msk = core.std.Expr([mm1, edm], "y x -").std.Binarize(20000).std.Maximum().std.Maximum()
ana = rektaa(src, 0, 62, 1378, 862, aatype=2, opencl=True, mtype=0)
ana = core.std.MaskedMerge(src, ana, msk)
snf = awf.rfs(snf, ana, "[16853 16900]")

ana = rektaa(src, 1248, 78, 520, 172, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[18198 18317] [18360 18431] [25630 25677]")

ana = rektaa(src, 770, 280, 1050, 486, aatype=2, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[20010 20285] [20400 20519]")

ana = rektaa(src, 598, 356, 816, 162, aatype=6, opencl=True, mtype=1)
ana = rekt(ana, src, 988, 540, 896, 476)
snf = awf.rfs(snf, ana, "[20682 20765]")

ana = rektaa(src, 0, 410, 886, 314, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[20766 20921] [21774 21881]")

ana = rektaa(src, 0, 356, 986, 308, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[20958 21005]")

# one moving darker line is affected
ana = rektaa(src, 352, aatype=6, opencl=True, mtype=2)
msk = y.std.Binarize(35000).std.Minimum().std.Minimum()
ana = ana.std.MaskedMerge(src, msk)
snf = awf.rfs(snf, ana, "[21092 21125]")

ana = rektaa(src, 0, 264, 1658, 0, aatype=3, opencl=True, mtype=1)
ana = rektaa(ana, 540, 118, 1066, 0, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[21882 21965]")

ana = rektaa(src, 238, 974, 1198, aatype=2, opencl=True, mtype=3)
ana = rektaa(ana, 820, 846, 968, 102, aatype=2, opencl=True, mtype=1)
ana = rektaa(ana, 440, 236, 1322, 436, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[21966 22025]")

ana = rektaa(src, 544, 154, 1298, 898, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[22062 22145]")

ana = rektaa(src, 1436, 980, 356, 22, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[23538 24149]")

ana = rektaa(src, 412, 116, 506, 586, aatype=6, opencl=True, mtype=1)
msk = y.std.Binarize(28000).std.Maximum().std.Inflate().std.Maximum()
ana = src.std.MaskedMerge(ana, msk)
snf = awf.rfs(snf, ana, "[24846 25013]")

# bunch of plants
ana = taa.TAAmbk(src, aatype=3, mtype=1, opencl=True)
msk = gmask(src, 110 << 8, 70 << 8, 100 << 8).std.Maximum().std.Inflate().std.Maximum()
ana = src.std.MaskedMerge(ana, msk)
snf = awf.rfs(snf, ana, "[25014 25123]")

# 24 fps rain and 12 fps objects with rest of the frame aliased
ana = taa.TAAmbk(src, aatype=2, mtype=3, opencl=True)
mmc = core.std.Expr([mm1, mm2], "x y +")
ana = ana.std.MaskedMerge(src, mmc)
snf = awf.rfs(snf, ana, "[25270 25313] [28612 28635] [52108 52142]")

ana = rektaa(src, 628, 310, 404, 472, aatype=6, opencl=True, mtype=3)
ana = src.std.MaskedMerge(ana, gmask(src, 90 << 8, 80 << 8, 110 << 8).std.Maximum().std.Maximum())
snf = awf.rfs(snf, ana, "[25678 25761]")

ana = rektaa(src, 820, 754, 978, 302, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[27000 27059]")

ana = rektaa(src, 878, 510, 390, 376, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[27060 27183]")

ana = rektaa(src, 542, 112, 474, 462, aatype=2, opencl=True, mtype=1)
msk = y.std.Binarize(23000).std.Maximum().std.Inflate().std.Maximum()
ana = ana.std.MaskedMerge(src, msk)
snf = awf.rfs(snf, ana, "[27232 27405]")

ana = rektaa(src, 1002, 236, 840, 750, aatype=2, opencl=True, mtype=1)
msk = y.std.Binarize(30000).std.Maximum().std.Inflate().std.Maximum()
ana = src.std.MaskedMerge(ana, msk)
snf = awf.rfs(snf, ana, "[27520 27555]")

ana = rektaa(src, 580, 468, 650, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[27556 27627]")

ana = rektaa(src, 0, 60, 1734, 416, aatype=6, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[27628 27693]")

ana = rektaa(src, 1038, 464, 762, 324, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[27868 27939]")

ana = rektaa(src, 520, 116, 1230, 902, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[28048 28123] [93878 93943]")

ana = rektaa(src, 1566, 244, 0, 692, aatype=2, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[33190 33216]")

ana = rektaa(src, 786, 0, 770, 948, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[33690 33809] [34056 34283] [34584 34649] [34974 35021]")

ana = rektaa(src, 36, 638, 1498, 382, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[37603 37685]")

ana = rektaa(src, 738, 730, 86, 228, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[42694 42945]")

ana = rektaa(src, 920, 20, 742, 994, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[44356 44463]")

ana = rektaa(src, 240, 0, 266, 0, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[44464 44511]")

ana = rektaa(src, 1282, 82, 354, 964, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[44548 44679] [45052 45099] [45184 45231]")

ana = rektaa(src, 1324, 880, 260, 134, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[45928 46047]")

ana = rektaa(src, 890, 18, 764, 990, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[46504 46575]")

ana = rektaa(src, 1064, 686, 0, 54, aatype=6, opencl=True, mtype=1)
ana = ana.std.MaskedMerge(src, mmc)
snf = awf.rfs(snf, ana, "[46576 46719]")# [47560 47799] [48844 48909]

ana = rektaa(src, 802, 568, 808, 438, aatype=3, opencl=True, mtype=1)
ana = rektaa(ana, 456, 790, 1326, aatype=2, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[47404 47559] [48052 48135] [48580 48651]")

ana = rektaa(src, 1378, 552, 346, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[48652 48711]")

ana = rektaa(src, 632, 128, 892, 502, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[48712 48771]")

ana = rektaa(src, 224, 0, 440, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[52883 52957]")

ana = rektaa(src,0, 166, 1858, 440, aatype=2, opencl=True, mtype=2)
ana = rektaa(ana, 746, 0, 704, 1032, aatype=2, opencl=True, mtype=1)
ana = rektaa(src, 114, 790, 1070, aatype=3, opencl=True, mtype=2)
ana = rektaa(ana, 330, 126, 1062, 806, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[53194 53278]")

ana = rektaa(src, 0, 0, 776, 928, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[53320 53381]")

ana = rektaa(src, 340, 352, 0, 44, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[53450 53497]")

ana = rektaa(src, 0, 298, 1808, 116, aatype=6, opencl=True, mtype=2)
ana = rektaa(ana, 1856, 62, 0, 134, aatype=6, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[55896 56243]")

ana = rektaa(src, 94, 880, 284, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[56568 56627]")

ana = rektaa(src, 1426, 748, 436, aatype=2, opencl=True, mtype=2)
ana = rekt(snf, ana, 1426, 748, 436)
snf = awf.rfs(snf, ana, "[56964 57023]")

ana = rektaa(src, 1006, 674, 516, 2, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[72090 72149]")

ana = taa.TAAmbk(src, aatype=6, opencl=True, mtype=2)
msk = gmask(src, 140 << 8, 60 << 8, 140 << 8).std.Minimum().std.Minimum().std.Maximum()
ana = ana.std.MaskedMerge(src, msk)
snf = awf.rfs(snf, ana, "[72318 72353]")

ana = rektaa(src, 256, 18, 910, 158, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[79358 79465]")

ana = rektaa(src, 1280, 274, 286, 198, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[84222 84281]")

ana = rektaa(src, 970, 296, 146, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[84852 84971]")

ana = rektaa(src, 1276, 474, 346, 436, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[85230 85481]")

ana = rektaa(src, 1086, 10, 692, 406, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[85884 85925]")

ana = rektaa(src, 254, 0, 218, 54, aatype=3, opencl=True, mtype=2)
ana = rekt(ana, rektaa(src, 378, 558, 602, 312, aatype=6, opencl=True, mtype=2), 378, 558, 602, 312)
snf = awf.rfs(snf, ana, "[90944 91069]")

ana = rektaa(src, 778, 508, 1104, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[91070 91225]")

ana = rektaa(src, 1048, 28, 820, 1038, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[91490 91531]")

ana = rektaa(src, 0, 778, 170, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[91604 91627]")

ana = rektaa(src, 330, 50, 1290, 506, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[92888 92923]")

ana = rektaa(src, 0, 454, 0, 554, aatype=3, opencl=True, mtype=1)
snf = awf.rfs(snf, ana, "[92924 92965]")

ana = rektaa(src, 28, 0, 1204, 12, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[92966 93001]")

ana = rektaa(src, 614, 894, 860, 0, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[94178 94225]")

ana = rektaa(src, 1372, 372, 108, 416, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[94226 94429]")

ana = rektaa(src, 356, 0, 244, 138, aatype=2, opencl=True, mtype=2)
ana = ana.std.MaskedMerge(src, y.std.Binarize(48000).std.Maximum().std.Maximum())
snf = awf.rfs(snf, ana, "[96632 96727]")

ana = rektaa(src, 290, 0, 1324, 40, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[99283 99354]")

ana = rektaa(src, 0, 542, 1314, aatype=3, opencl=True, mtype=2)
ana = rektaa(ana, 1436, 534, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[99379 99522]")

ana = rektaa(src, 236, 48, 636, 112, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[99697 99936]")

ana = rektaa(src, 728, 30, 0, 330, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[99967 99990]")

ana = rektaa(src, 0, 0, 1648, 422, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[100063 100146]")

ana = rektaa(src, 512, 328, 1006, 482, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[104650 104823]")

ana = rektaa(src, 512, 328, 954, 482, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[105283 105389] [105528 105707]")

ana = rektaa(src, 782, 478, 0, 74, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[110216 110299]")

ana = rektaa(src, 1118, 122, 18, 26, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[114244 114315]")

ana = rektaa(src, 352, 40, 1350, 400, aatype=6, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[114376 114597]")

ana = rektaa(src, 616, 0, 12, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[114922 115029]")

ana = rektaa(src, 616, 598, 966, 130, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[115174 115257] [115414 115497]")

ana = rektaa(src, 936, 20, 428, 242, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[115942 116013]")

ana = rektaa(src, 898, 178, 246, 270, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[118263 118304] [119711 119782]")

ana = rektaa(src, 272, 242, 1414, 520, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[120503 120628]")

ana = rektaa(src, 550, 278, 938, 714, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[145671 145718]")

ana = rektaa(src, 568, 342, 784, 516, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[145719 145802]")

ana = rektaa(src, 746, 850, 488, aatype=2, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[146394 146606]")

ana = rektaa(src, 576, 288, 436, 492, aatype=3, opencl=True, mtype=2)
snf = awf.rfs(snf, ana, "[147045 147104]")

# banding
dbn = src.neo_f3kdb.Deband(y=72, cb=64, cr=64, grainy=0, grainc=0, output_depth=16, sample_mode=4)
msk = kgf.retinex_edgemask(src).std.Maximum().std.Inflate()
dbn = dbn.std.MaskedMerge(src, msk)
dbn = adptvgrnMod(dbn, static=True, grain_chroma=True, strength=.4, seed=7)
snf = awf.rfs(snf, dbn, "[57480 58055] [26558 26609]")

# weak banding
dbn = src.neo_f3kdb.Deband(y=64, cb=64, cr=64, grainy=0, grainc=0, output_depth=16, sample_mode=4)
dbn = dbn.std.MaskedMerge(src, msk)
dbn = adptvgrnMod(dbn, static=True, grain_chroma=True, strength=.3, seed=7)
snf = awf.rfs(snf, dbn, "[31406 31513] [90272 90511] [96541 96631] [117456 117533]")

# misplaced frame
snf = snf.std.FreezeFrames(45771, 45771, 45770)

out = depth(snf, 8)

out = adptvgrnMod(out, strength=.1, static=True, luma_scaling=100, seed=8)

out.set_output()
