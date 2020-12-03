import vapoursynth as vs
import vsutil
import dysfunctional
import havsfunc
import vsTAAmbk as taa
import awsmfunc
from rekt import rekt, rekt_fast
import lvsfunc

core = vs.core

def adaptive_edgemask(clip: vs.VideoNode, edgemask, pre=None, modifier=None, luma_scaling=12, invert=True, **args) -> vs.VideoNode:
    from vsutil import iterate, get_y, depth, Range
    
    b32 = depth(get_y(clip), 32, range_in=Range.LIMITED, range=Range.LIMITED)

    if pre != None:
        prefilter = pre(b32) 
        edgem = edgemask(prefilter, **args)
    else: edgem = edgemask(b32, **args)
    
    if modifier != None:
        edgem = iterate(edgem, lambda x: core.std.Minimum(x), modifier[0])
        edgem = iterate(edgem, lambda x: core.std.Maximum(x), modifier[1])
        edgem = iterate(edgem, lambda x: core.std.Inflate(x), modifier[2])
        
    adptv = core.adg.Mask(core.std.PlaneStats(b32), luma_scaling)
    if invert is True: adptv = core.std.Invert(adptv)

    merge = core.std.Expr([adptv, edgem], ["x y +"])
    return depth(merge, clip.format.bits_per_sample, dither_type='none', range=Range.FULL, range_in=Range.FULL)

# lvsfunc.scale.descale(vsutil.depth(src, 32), height=855, kernel=lvsfunc.kernels.Bicubic(b=0, c=1/2), threshold=0.0, show_mask=False)
src = core.lsmas.LWLibavSource(r'00000.m2ts')
des = core.lsmas.LWLibavSource(r'GRAY16Intermediary.mkv')
mrg = core.std.ShufflePlanes([des, vsutil.depth(src, 16, dither_type='none')], [0,1,2], vs.YUV)

msk_1 = adaptive_edgemask(mrg, lambda x: dysfunctional.FDOG(x), modifier=[0,0,0], invert=True, luma_scaling=1)
msk_2 = dysfunctional.FDOG(mrg, retinex=False, div=[3,2])

# Overlay borders from pre-descaled source
b16 = vsutil.depth(src, 16)
dln = rekt(mrg, b16, left=b16.width -4)
dln = rekt(dln, b16, right=b16.width -4)
dln = rekt(dln, b16, top=b16.height -4)
dln = rekt(dln, b16, bottom=b16.height -4)

dln_var = awsmfunc.FillBorders(dln, planes=0, left=1)
dln_var = awsmfunc.rfs(dln, dln_var, "[42 434] [963 1622] [2032 2472] [2509 2581] [2617 2907] [3013 3272] [3321 3422] [3546 4154] [4323 5443] [5492 8029] [8438 8869]")

dln_var2 = awsmfunc.bbmod(dln_var, left=3, right=3, y=False)
dln_var2 = awsmfunc.rfs(dln_var, dln_var2, "[154 404]")

dln = awsmfunc.bbmod(dln_var2, scale_thresh=True, thresh=15, blur=40, left=2, top=2, right=2, bottom=1)
dln = awsmfunc.rfs(dln, mrg, "[2908 3012]")

# Fix poor dither (?) with KNL
# I was thinking that perhaps if I could generate an artificial pattern I could use do some magic with diffs
# ... but I never tried (probably wouldn't work given all the excess noise anyway)
knl = vsutil.depth(core.knlm.KNLMeansCL(vsutil.depth(dln, 32), h=1.3, channels='Y'), 16)
bmsk = core.std.Binarize(vsutil.get_y(dln), 185 << 8, planes=0).std.Deflate()

grn = core.grain.Add(knl, .2, seed=444, constant=True)
wtf_1 = core.std.MaskedMerge(grn, dln, msk_2)
wtf_1 = awsmfunc.rfs(dln, wtf_1, "[435 962] [3423 3545] [8030 8437]")

wtf_2 = core.std.MaskedMerge(dln, grn, bmsk)
wtf_2 = awsmfunc.rfs(wtf_1, wtf_2, "[1623 1796] [2908 3012] [3273 3320] [4155 4322] [5444 5491]")

shf = wtf_2

dbn = core.f3kdb.Deband(shf, preset='luma', y=80, grainy=45)
dbn = core.std.MaskedMerge(dbn, shf, msk_2)
dbn = awsmfunc.rfs(shf, dbn, "[4690 4785]")

dbn_2 = core.f3kdb.Deband(shf, preset='luma/nograin', y=50)
dbn_2 = core.grain.Add(dbn_2, .5, seed=444)
dbn_2 = core.std.MaskedMerge(dbn_2, dbn, msk_2)
dbn_2 = awsmfunc.rfs(dbn, dbn_2, "[7196 7573]")

aa = taa.TAAmbk(dbn_2, 1, 0, 0, opencl=True)
aa = awsmfunc.rfs(dbn_2, aa, "[435 1058] [1530 3545] [3642 5115] [5338 7573] [8438 8869]")

aa_2 = rekt_fast(aa, fun=lambda x: lvsfunc.aa.upscaled_sraa(x), top=480, bottom=340, left=690, right=1040)
aa_2 = awsmfunc.rfs(aa, aa_2, "[1059 1322]")

aa_3 = rekt_fast(aa, fun=lambda x: lvsfunc.aa.upscaled_sraa(x), top=300, bottom=500, left=1550)
aa_3 = awsmfunc.rfs(aa_2, aa_3, "[4690 4785]")

aa_4 = lvsfunc.aa.upscaled_sraa(aa_3)
aa_4 = awsmfunc.rfs(aa_3, aa_4, "[5116 5325]")

pln = rekt_fast(aa_4, fun=lambda m: lvsfunc.aa.upscaled_sraa(m, rfactor=1.4, gamma=10, alpha=.5, beta=.5), top=730, bottom=304, left=440, right=100)
pln = awsmfunc.rfs(aa_4, pln, "[7574 8029]")

ple = rekt_fast(pln, fun=lambda m: lvsfunc.aa.upscaled_sraa(lvsfunc.aa.upscaled_sraa(m, rfactor=1.2)), top=0, bottom=0, left=800, right=1020)
ple = awsmfunc.rfs(pln, ple, "[2908 3012]")

# Probably an issue from the master, causing pink line art
# Fixed by desaturating, applied only to line art in overlapping rectangles
# Tcanny -1 is gauss blur, for sobel.
gus = core.tcanny.TCanny(vsutil.get_y(ple), .5, mode=-1, planes=0)
lne = core.std.Sobel(gus, planes=[0]).std.Maximum()

dst = awsmfunc.saturation(ple, 0.2)
cma = core.std.MaskedMerge(ple, dst, lne)
crp = rekt(ple, cma, top=482, bottom=494, right=1732, left=72)
fix = rekt(crp, ple, left=172, bottom=508, right=1722)
fix = awsmfunc.rfs(ple, fix, "[419 424]")

# I'm not doing this manually
hlo = havsfunc.FineDehalo(fix, darkstr=0, brightstr=.7, showmask=0)

out = vsutil.depth(hlo, 8)
out.set_output()