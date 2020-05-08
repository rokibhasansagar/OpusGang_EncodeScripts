import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import mvsfunc as mvf
import kagefunc as kgf
from adptvgrnMod import *
    
src = core.ffms2.Source("")

src = core.std.Crop(src, 0, 0, 22, 22)

src = fvf.Depth(src, 16)

bb=core.edgefixer.ContinuityFixer(src, left=1, top=[3, 4, 4], right=0, bottom=[4, 2, 2], radius=10)
fb = core.fb.FillBorders(bb, left=2)
bb = core.std.Merge(bb, fb, [0, 1])
sf=fvf.ReplaceFramesSimple(src,bb,mappings="[0 359]")

bb=core.edgefixer.ContinuityFixer(src, left=2, top=[4, 4, 4], right=0, bottom=0, radius=10)
sf=fvf.ReplaceFramesSimple(sf,bb,mappings="[19081 19114]")

bb=core.edgefixer.ContinuityFixer(src, left=0, top=5, right=0, bottom=0,radius=10)
fb=core.fb.FillBorders(bb,left=0,right=0,top=5,bottom=0,mode="fillmargins")
merge=core.std.Merge(bb,fb,[0,1])
sf=fvf.ReplaceFramesSimple(sf,merge,mappings="[45655 45701] [45776 45858]")

dn = core.knlm.KNLMeansCL(src, h=0.8)
dndiff = core.std.MakeDiff(src, dn)
db = core.f3kdb.Deband(dn, range=25, y=64, cb=0, cr=0, grainy=32, grainc=0, output_depth=16, keep_tv_range=True)
mask = kgf.retinex_edgemask(src).std.Expr("x 10000 > x 2 * x 0.8 * ?").std.Inflate()
mergem = core.std.MaskedMerge(db, src, mask)
merge = core.std.MergeDiff(mergem, dndiff)
merge = adptvgrnMod(merge, size=3.5, strength=0.2, sharp=50, luma_scaling=1, grain_chroma=False, static=False)
sf = fvf.rfs(sf, merge, mappings="[1148 1277] [32980 33036] [50995 51045] [66689 66805] [76171 76288] [76876 77053] [77954 78141] [78227 78339] [80169 80185] [80329 80348] [80732 80753] [80846 80997] [80998 81030] [81387 81506] [85332 85413] [86600 86675] [91853 92058] [92059 92242] [92278 92324] [92577 92619] [97387 97415] [97416 97467] [97600 97674] [97704 97738] [100466 100574] [100575 100646] [103056 103199] [103506 103546] [105259 105364] [119478 119635] [123648 123805] [128279 128335] [129699 129754] [131435 131454] [131473 131552] [131587 131636] [132028 132130] [132932 133282] [133898 133969] [136293 138519] [138573 139934] [140124 140227] [140362 141120] [141188 141358] [141403 142166] [142285 142400] [142482 142927] [143244 143527] [143973 144186] [144543 144914] [145281 145316] [145557 145714] [145834 146042] [146319 146602] [147539 147576] [147795 147936] [147974 148124] [148253 148349] [148386 148505] [148542 149036] [149171 149244] [149359 149473] [149658 149710] [149711 149829] [150148 150188] [151898 151959] [151990 152037] [154786 154808] [154839 154873] [154924 154971] [155122 155223] [156977 157096] [157144 157279] [157325 157380] [158762 158835] [158863 158926] [159440 159466] [161897 162029] [162169 162242] [162766 163127] [163489 163638] [164476 164563] [165677 165786] [165894 165948] [166853 167013] [167109 167174] [169002 169164] [171044 171110] [171156 171197] [173041 173100] [173193 173474] [173542 173715] [173895 174126] [174195 174301] [175463 175483] [176538 176596] [177690 177785] [180670 180733] [35408 35541] [40451 40652] [41401 41487] [44144 44223] [45655 45701] [45776 45858] [50525 50589] [68306 68379] [70219 70380] [70961 70998]")

out = mvf.Depth(sf, 10, dither=7)

out.set_output()