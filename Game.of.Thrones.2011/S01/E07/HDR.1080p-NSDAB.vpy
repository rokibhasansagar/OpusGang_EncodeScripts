import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
import vsTAAmbk as taa
import dfmderainbow

core.max_cache_size=15000

src=core.ffms2.Source("")
e06 = core.ffms2.Source("")
e06 = fvf.Debilinear(fvf.Depth(e06, 16), 1920, 1080, yuv444=True)

src=fvf.Depth(src,16)

src=fvf.Debilinear(src,1920,1080,yuv444=True)

dr=dfmderainbow.DFMDerainbow(src,10)

aa=taa.TAAmbk(dr, aatype=2, aatypeu=6, aatypev=6, opencl=True, mtype=1, mpand=[0, 0], down8=False, aarepair=0)
aa1=taa.TAAmbk(dr, aatype=6, aatypeu=6, aatypev=6, opencl=True, mtype=1, mpand=[0, 0], down8=False, aarepair=24)

db = src.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(src)
merged = core.std.MaskedMerge(db, src, mask)
sf=fvf.ReplaceFramesSimple(aa, merged, mappings="[135 191]")

db=aa.f3kdb.Deband(range=20,y=10,cb=10,cr=10,grainy=32,grainc=32,output_depth=16,dynamic_grain=True)
merged=core.std.MaskedMerge(db,aa,mask.std.Binarize(5700))
sf=fvf.rfs(sf,merged,"[2781 2804]")

text=taa.TAAmbk(src,aatype=3,aatypeu=0,aatypev=0,mtype=1)
sf=fvf.rfs(sf,text,"[192 2780] [81588 83197]")

# swapping out intro to that from previous episode
sf = fvf.rfs(sf, e06, "[0 422] [472 667] [771 777] [893 954] [1006 2602]")

final=fvf.Depth(sfdb,10)

final.set_output()