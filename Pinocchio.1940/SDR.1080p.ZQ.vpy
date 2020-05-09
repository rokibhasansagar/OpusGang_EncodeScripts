import vapoursynth as vs
import awsmfunc as awsm
import kagefunc as kgf
import fvsfunc as fvf
import adptvgrnMod as agm
import havsfunc as hvf
import vsTAAmbk as taa
import debandshit as dbs
import lvsfunc as lvf
from havsfunc import *
from vsutil import *
from rekt import *
from bandmask import *
core = vs.core

src = core.ffms2.Source(r'Pinocchio 1940 1080p Remux AVC FLAC 5.1.mkv')
src = core.std.Crop(src, left = 234, right = 234, top = 2)

b16 = fvf.Depth(src, 16)
dln = awsm.bbmod(b16, left = 2, right = 1)

lmsk = core.std.Binarize(core.std.Maximum(core.std.Maximum(kgf.retinex_edgemask(dln))),42000)
lmsk_2 = kgf.retinex_edgemask(dln)
dmsk = core.std.Maximum(core.std.Sobel(dln, [0,1,2], 1.4))
adptvmsk = core.adg.Mask(core.std.PlaneStats(dln), 25).std.Invert()
adptvmsk_2 = core.adg.Mask(core.std.PlaneStats(dln), 30).std.Invert()
gmsk_pre = core.knlm.KNLMeansCL(dln, h = .4, channels = 'Y')
gmsk = bandmask(dln, 300, dec = 4).std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Maximum().std.Invert()
gmsk_2 = bandmask(dln, 800, dec = 27).std.Invert()
gmsk_3 = bandmask(dln, 450).std.Invert()
gmsk_4 = bandmask(dln, 550, dec = 25).std.Invert()
gmsk_5 = bandmask(dln, 290, dec = 3).std.Invert()
fmsk = dbs.lumamask(dln, 4096, 8092, False)
peak = (1 << dln.format.bits_per_sample) - 1
emsk = core.std.Sobel(dln,[0], 0.4).std.Expr(expr=[f'x {scale(4, peak)} < 0 x {scale(32, peak)} > {peak} x ? ?']).std.Invert().std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1]).std.Deflate().std.Minimum().std.Minimum().std.Minimum().std.Minimum()
hmsk = dbs.lumamask(dln, 7000, 10000, False)

aa = taa.TAAmbk(dln, 1, mtype = 1, opencl = True)
aa = fvf.rfs(dln, aa, "[36461 36533] [40991 40991]")

dn = aa

f3k1 = agm.adptvgrnMod(core.f3kdb.Deband(dn, output_depth = 16, preset = 'low/nograin'), strength = .2, luma_scaling = 25, size = 1.35, sharp = 47, static = False, show_mask=False)
f3k2 = agm.adptvgrnMod(core.f3kdb.Deband(dn, output_depth = 16, preset = 'medium/nograin', y = 55), strength = .45, cstrength = .18, luma_scaling = 0, size = 1.7, sharp = 45, static = False)
def f3k3(clip):
    f3k3 = core.knlm.KNLMeansCL(clip, h = 5, s = 4, device_type = 'gpu')
    f3k3 = core.knlm.KNLMeansCL(f3k3, h = 3, channels = 'UV', device_type = 'gpu')
    f3k3 = core.f3kdb.Deband(f3k3, output_depth = 16, preset = 'nograin', y = 80, cb = 150, cr = 100)
    f3k3 = agm.adptvgrnMod(f3k3, strength = .9, luma_scaling = 12, size = 2.25, sharp = 35, static = False)
    return f3k3
f3k3 = f3k3(dn) # This is dumb
f3k4 = agm.adptvgrnMod(core.f3kdb.Deband(dn, output_depth = 16, preset = 'medium/nograin', cb = 32, cr = 32), strength = .2, cstrength = .18, luma_scaling = 0, size = 1.7, sharp = 45, static = False)
f3k5 = agm.adptvgrnMod(core.f3kdb.Deband(dn, output_depth = 16, preset = 'medium/nograin'), strength = 1.25, size = 1.4, sharp = 30, static = False)
f3k6 = agm.adptvgrnMod(core.f3kdb.Deband(dn, output_depth = 16, preset = 'high/nograin'), strength = .65, cstrength = .18, luma_scaling = 0, size = 1.7, sharp = 45, static = False)
f3k7 = agm.adptvgrnMod(core.f3kdb.Deband(dn, output_depth = 16, preset='nograin', y = 40, cb = 24, cr = 24), strength = .32, cstrength = .18, luma_scaling = 0, size = 1.7, sharp = 45, static = False)
thxlypheo = fvf.Depth(core.placebo.Deband(fvf.Depth(dln, 32), dither=False, planes=1|2|4, grain=8, radius=24, threshold=4), 16)

db1 = core.std.MaskedMerge(f3k1, dn, lmsk)
db1 = core.remap.Rfs(dn, db1, mappings = "[2154 2606] [2934 3391] [3988 4528] [4754 4866] [12162 12162] [37492 37495]")

db2 = core.std.MaskedMerge(f3k2, db1, gmsk)
db2 = core.remap.Rfs(db1, db2, mappings = "[8233 8329] [8517 8593] [8670 8708] [9285 9348] [9518 9821] [10422 11241] [11732 11841] [12062 12161] [12282 13072] [13159 13552] [14869 15022] [15287 15359] [15523 15666] [15743 15874] [15972 16023] [17518 17567] [17624 17869] [17981 18426] [18556 18785] [19190 19309] [19354 19441] [19571 19640] [19800 20071] [20150 20423] [20582 20647] [20993 21040] [21848 21905] [24188 24221] [24706 24786] [27371 27455] [27531 27578] [27625 27713] [27933 28027] [28239 28508] [28565 28611] [28717 28801] [28886 29036] [29143 29184] [29301 29445] [29489 29604] [30711 30742] [30916 31003] [31047 31051] [31131 31325] [31409 31478] [31690 32477] [32526 32598] [32831 32995] [33248 33447] [33537 33617] [33755 33931] [34006 34200] [34248 34658] [34707 34923] [34968 35175] [35176 36290] [36407 36691] [36736 37018] [37107 37218] [37247 37490] [37496 38500] [38961 38979] [38996 39019] [40040 41191] [41268 41423] [41568 42045] [42177 42225] [42367 42498] [42553 42704] [42898 43483] [43676 43733] [43754 44725] [46982 47517] [47753 48978] [49188 49188] [49303 50523] [50621 50749] [50945 51047] [51509 51823] [52050 52473] [52718 52985] [53106 53393] [54034 54121] [54658 55611] [55850 56981] [57162 57337] [58001 58156] [58334 58595] [59069 63645] [66208 66271] [66386 66686] [68650 68804] [73176 76045] [77099 78696] [78982 79110] [79199 79421] [79518 81427] [81804 81985] [82724 83143] [83236 83517] [83679 83787] [83867 83945] [85811 85811] [86019 86100] [86391 86423] [86664 87065] [87492 87698] [87860 87947] [88088 88192] [88396 88434] [88551 88687] [89668 89692] [89981 90131] [90826 90984] [91065 91832] [91952 92120] [92233 92428] [92516 94015] [94268 94311] [94356 94534] [94587 94678] [94929 94969] [95034 95063] [96064 96119] [97697 97767] [98294 98325] [98406 98453] [98559 98603] [98866 98942] [99146 99280] [102419 103455] [103978 104168] [104252 104410] [104446 104869] [105096 105287] [106172 106405] [106560 107529] [108190 108501] [108805 108817] [108861 108964] [109114 109346] [109406 109541] [109614 109701] [110800 110871] [111029 111052] [111248 111871] [111944 113311] [113552 113656] [113896 115287] [116470 116722] [117083 117236] [117601 117775] [117892 117980] [118092 118247] [118429 118621] [118985 119271] [120918 121125] [121551 121805] [122597 123154] [123201 123265] [123303 123528] [124280 124367] [124868 125077] [125603 126165]")

db3 = core.std.MaskedMerge(f3k3, db2, adptvmsk)
db3 = core.remap.Rfs(db2, db3, mappings = "[18930 18997]")

db4 = core.std.MaskedMerge(f3k4, db3, gmsk_2)
db4 = core.remap.Rfs(db3, db4, mappings = "[38501 38572] [57558 57594] [57600 57637] [58886 59025] [72688 72706] [84525 84846] [94694 94734] [101720 101996]")

db5 = core.std.Merge(f3k5, db4)
db5 = core.remap.Rfs(db4, db5, mappings = "[38803 38858]")

db6 = core.std.MaskedMerge(f3k6, db5, gmsk_3)
db6 = core.remap.Rfs(db5, db6, mappings = "[44736 46125] [119273 119343] [119425 119495]")

db7 = core.std.MaskedMerge(f3k2, db6, hmsk)
db7 = core.remap.Rfs(db6, db7, mappings = "[49189 49241] [51272 51508] [65120 65962] [67873 67993] [68160 68231] [68384 68511] [76442 76574] [76637 76687] [83788 83866] [84144 84287] [87234 87297] [87647 87859] [88193 88395] [95064 95149] [97367 97567] [99282 99353] [100640 101110] [102001 102418] [104411 104445] [105288 105426] [108550 108579] [108590 108645] [109542 109613] [109702 110093] [116847 116958] [121947 122042] [123828 123967]")

db8 = core.std.MaskedMerge(f3k2, db7, gmsk_4)
db8 = core.remap.Rfs(db7, db8, mappings = "[95336 95415] [99608 99687]")

db9 = core.std.MaskedMerge(thxlypheo, db8, gmsk_5)
db9 = core.remap.Rfs(db8, db9, mappings = "[98971 99145] [108819 108823]")

db10 = core.std.MaskedMerge(f3k4, db9, adptvmsk_2)
db10 = core.remap.Rfs(db9, db10, mappings = "[105984 106154]")

db11 = core.std.MaskedMerge(f3k7, db10, gmsk_2)
db11 = core.remap.Rfs(db10, db11, mappings = "[108818 108860] [109059 109113] [110148 110349] [113856 113895]")

final = fvf.Depth(db11, 8)
final.set_output()