import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import lvsfunc as lvf
import awsmfunc as awf
from vsutil import *
import dfmderainbow
from adptvgrnMod import *
import vsTAAmbk as taa

core.max_cache_size=1024

src = core.ffms2.Source("Game.of.Thrones.S01E07.You.Win.or.You.Die.2160p.UHD.BluRay.REMUX.HDR.HEVC.Atmos-EPSiLON.mkv")

src = depth(src,16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

dr = dfmderainbow.DFMDerainbow(src, 10)

aa = lvf.sraa(dr, rfactor=1.8)
ref = get_y(src).std.Convolution(9 * [1])
ref = kgf.kirsch(ref).std.Binarize(13000).std.Maximum()
aa = src.std.MaskedMerge(aa, ref)

dbn = src.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
msk = kgf.retinex_edgemask(src)
mrg = core.std.MaskedMerge(dbn, src, msk)
src = awf.rfs(aa, mrg, mappings="[135 191]")

dbn = aa.neo_f3kdb.Deband(range=20, y=48, cb=48, cr=48, grainy=0, grainc=0, output_depth=16, sample_mode=4)
msk = msk.std.Binarize(8000).std.Maximum()
dbn = dbn.std.MaskedMerge(aa, msk)
bms = awf.bandmask(src, thr=100).std.Inflate().std.Maximum().std.Inflate()
dbn = aa.std.MaskedMerge(dbn, bms)
agm = adptvgrnMod(aa, show_mask=True, luma_scaling=50)
dbn = sizedgrn(dbn, strength=.3, cstrength=.1, sharp=90, size=1.2)
dbn = aa.std.MaskedMerge(dbn, agm)
src = awf.rfs(src, dbn, "[226 77977]")

txt = taa.TAAmbk(src, aatype=3, aatypeu=0, aatypev=0, mtype=1)
src = awf.rfs(src, txt, "[192 2780] [81588 83197]")

src = depth(src, 10)
