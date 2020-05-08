import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import mvsfunc as mvf
import kagefunc as kgf
import havsfunc as hvf
import vsTAAmbk as taa
import fag3kdb
import nnedi3_resample as nnrs
import BalanceBorders
import sgvsfunc as sgf

src=core.ffms2.Source("")

#progress 32124

gbr=core.std.Trim(src,0,23)+(core.ffms2.Source(""))
ger=fvf.Depth(src,16)
gbr=fvf.Depth(gbr,16)
dehardsub=bzf.dehardsub(ger,gbr)
dehardsub=fvf.Depth(dehardsub,8)
src=fvf.ReplaceFramesSimple(src,dehardsub,mappings="[9743 9805] [12163 12205] [12277 12339] [31732 31787] [32098 32123]")

bb = core.edgefixer.ContinuityFixer(src, left=3, top=3, right=3, bottom=3, radius=10)
bb=fvf.ReplaceFramesSimple(src,bb,mappings="[56 191] [29142 29202]")

b16 = mvf.Depth(bb, 16)

dbfk = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=32, grainc=32, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
merged = core.std.MaskedMerge(dbfk, b16, mask)
merged = mvf.Depth(merged, 8)
sfdbfk=fvf.ReplaceFramesSimple(bb, merged, mappings="[457 564] [573 2367] [2397 2690] [22291 22293] [22312 22314] [22784 22786] [22792 22794] [22929 22931] [22937 22978] [32090 32123]")

dbfks = b16.f3kdb.Deband(range=20, y=100, cb=100, cr=100, grainy=10, grainc=10, output_depth=16)
masks = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
mergeds = core.std.MaskedMerge(dbfks, b16, masks)
mergeds = mvf.Depth(mergeds, 8)
sfdbfks=fvf.ReplaceFramesSimple(sfdbfk, mergeds, mappings="")

dbfks2 = b16.f3kdb.Deband(range=20, y=70, cb=60, cr=60, grainy=32, grainc=32, output_depth=16, dynamic_grain=True)
masks2 = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
mergeds2 = core.std.MaskedMerge(dbfks2, b16, masks2)
mergeds2 = mvf.Depth(mergeds2, 8)
sfdbfks2=fvf.ReplaceFramesSimple(sfdbfk, mergeds2, mappings="[2397 2690] [4509 4551] [4717 4758] [5634 5705] [22257 22263] [22268 22269] [22278 22280] [22284 22290] [22300 22304] [22308 22311] [22323 22780] [22799 22928] [22979 23349] [23710 23771] [25220 25541] [25558 25574] [25891 26081] [26265 26334] [26457 26654] [26688 26797] [27114 27190] [27424 27754] [27902 27928] [28015 28837] [28855 29039] [29069 29275] [29511 29641] [31788 31917]")

dbf = b16.f3kdb.Deband()
dbf8 = mvf.Depth(dbf,8)
sfdbf=fvf.ReplaceFramesSimple(sfdbfks2, dbf8, mappings="")

aa = taa.TAAmbk(sfdbf,aatype=-3,preaa=-1,strength=-0.5,mtype=2)
sfaa=fvf.ReplaceFramesSimple(sfdbf,aa,mappings=" [793 923] [938 1026] [1207 1484] [2154 2367]")

aa2=taa.TAAmbk(sfaa,aatype=-3,preaa=-1,strength=0.4,mtype=2)
sfaa2=fvf.ReplaceFramesSimple(sfaa,aa2,mappings="")

aa3=taa.TAAmbk(sfaa2,aatype=-3,preaa=-1,strength=0.2,mtype=2)
sfaa3=fvf.ReplaceFramesSimple(sfaa2,aa3,mappings="")

sfaa3.set_output()