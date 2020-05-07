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

src=core.ffms2.Source(r"")

crop=core.std.Crop(src,top=42,bottom=42)
crop=fvf.DebicubicM(crop,1998,1080)
crop=core.resize.Spline36(crop,1920,1038)

b16 = mvf.Depth(crop, 16)

b16=hvf.FineDehalo(b16,rx=2.0,darkstr=0.0)

dbfk = b16.f3kdb.Deband(range=15, y=60, cb=0, cr=0, grainy=22, grainc=0, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Binarize(3000).std.Maximum().std.Inflate()
merged = core.std.MaskedMerge(dbfk, b16, mask)
merged = mvf.Depth(merged, 16)
sfdbfk=fvf.ReplaceFramesSimple(b16, merged, mappings="[34259 34387] [79905 80039] [80286 80573] [80694 81110] [111752 111889] [112109 112252]")

final=mvf.Depth(sfdbfk,10)

final.set_output()