import vapoursynth as vs
core = vs.core
core.max_cache_size = 1024
import awsmfunc as awf
from vsutil import *
import vsTAAmbk as taa
from adptvgrnMod import *
import dysfunctional as dyf
import havsfunc as haf
import lvsfunc as lvf
import havsfunc as haf

src = core.ffms2.Source("abyss.mkv")

src = depth(src, 16)

# eedi3 for harsher aliasing, limit it to areas nnedi3 doesn't affect
# then perform a low and high pass to limit change

y = get_y(src)
eaa = taa.TAAmbk(y, aatype=2, mtype=0, down8=0)
naa = taa.TAAmbk(y, aatype=3, mtype=0, down8=0) # test down8=0

# brighten, darken, nnedi3-source difference
thr = [800, 1500, 500]

expr_n = f"y x - {thr[0]} > x {thr[0]} + y x - -{thr[1]} < x {thr[1]} - y ? ?"
expr_e = expr_n.replace("y", "z")
naa = core.std.Expr([y, naa, eaa], f"y z - abs {thr[2]} > {expr_e} {expr_n} ?")

# weaker TCanny mask since most lines are fine
ems = y.tcanny.TCanny(sigma=.5, t_h=6, t_l=3, mode=0).std.Maximum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1]).std.Deflate()

y = y.std.MaskedMerge(naa, ems)

naa = core.std.ShufflePlanes([y, src], [0, 1, 2], vs.YUV)
naa = awf.rfs(src, naa, "[741 143651]")

# horrible aliasing
saa = lvf.sraa(src, 1.2, 17)
saa = src.std.MaskedMerge(saa, ems)
src = awf.rfs(naa, saa, "[114858 114866]")

mnl = core.imwri.Read("manual.png").std.AssumeFPS(src)
mnl = mnl.resize.Point(format=vs.GRAY16, dither_type="error_diffusion", matrix_s="709")
sed = taa.TAAmbk(mnl, aatype=-2, strength=.5, mtype=0)
dif = core.std.Expr([mnl, sed], "x y - abs 600 > 65535 0 ?")
msk = core.std.Expr([dif, mnl.std.Sobel().std.Binarize(12000)], "x y = x 0 ?").std.Maximum().std.Maximum()
mnl = mnl.std.MaskedMerge(sed, msk)
mnl = core.std.ShufflePlanes([mnl, src], [0, 1, 2], vs.YUV)
src = awf.rfs(src, mnl, "114905")

# garbage deblock stolen from autodeblock lol
dbl = haf.Deblock_QED(src.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2), quant1=28, quant2=30, aOff1=3, bOff1=3, aOff2=3, bOff2=3, uv=2)
dbl = dbl.dfttest.DFTTest(sigma=.9, tbsize=1, planes=[0, 1, 2])
dbl = lvf.sraa(dbl)
dbl = sizedgrn(dbl, static=True, strength=.4, cstrength=.3, seed=7171)
src = awf.rfs(src, dbl, "[114969 115018] [129230 129275]")

# simple debanding
dbn = src.neo_f3kdb.Deband(range=25, y=64, cb=64, cr=64, grainy=0, grainc=0, sample_mode=4)
dbn = sizedgrn(dbn, strength=.2, static=True, seed=7070)
msk = awf.bandmask(src, thr=100).std.Maximum()
dbn = src.std.MaskedMerge(dbn, msk)
src = awf.rfs(src, dbn, "[54979 55008] [144732 144958]")

# certain red textures
lms = y.std.Expr("x 85 > x 88 < and x 106 > x 109 < and or 65535 0 ?")
dbl = src.std.MaskedMerge(dbn, lms)
src = awf.rfs(src, dbl, "[67639 67769] [69810 69857]")

# debanding with an edge mask wow
ems = dyf.FDOG(y, sigma=1.5).std.Expr("x 5000 > x 3 * x ?").std.Inflate().std.Inflate()
dbe = dbn.std.MaskedMerge(src, ems)
src = awf.rfs(src, dbe,  "[67836 67860] [105521 105581] [105700 105761] [113668 113756] [114598 114918] [115026 115033]")

# grain brights and darks yikes
grn = sizedgrn(src, strength=.3, cstrength=.1, static=True, seed=6969)
ms1 = adptvgrnMod(src, show_mask=True, luma_scaling=100)
ms2 = adptvgrnMod(src.std.Invert(), show_mask=True, luma_scaling=100)
src = src.std.MaskedMerge(grn, ms1).std.MaskedMerge(grn, ms2)

src = depth(src, 8)
  
src.set_output()