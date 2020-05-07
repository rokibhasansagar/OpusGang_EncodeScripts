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

src=core.std.Crop(src,top=138,bottom=138)
bb=sgf.FixColumnBrightnessProtect2(src,0,0)

b16 = mvf.Depth(bb, 16)

dbfk = b16.f3kdb.Deband(range=15, y=50, cb=50, cr=50, grainy=15, grainc=15, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
merged = core.std.MaskedMerge(dbfk, b16, mask)
merged = mvf.Depth(merged, 8)
sfdbfk=fvf.ReplaceFramesSimple(bb, merged, mappings="[27 317]")

dbfks = b16.f3kdb.Deband(range=20, y=64, cb=0, cr=0, grainy=32, grainc=0, output_depth=16,dynamic_grain=True)
masks = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
mergeds = core.std.MaskedMerge(dbfks, b16, masks)
mergeds = mvf.Depth(mergeds, 8)
sfdbfks=fvf.ReplaceFramesSimple(sfdbfk, mergeds, mappings="[318 551] [22983 23112] [23248 23346] [25198 25383] [25491 25707] [27410 27550] [30199 30309] [30350 30558] [34303 34544] [102635 102699]")

dbfks2 = b16.f3kdb.Deband(range=20, y=70, cb=60, cr=60, grainy=15, grainc=0, output_depth=16)
masks2 = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
mergeds2 = core.std.MaskedMerge(dbfks2, b16, masks2)
mergeds2 = mvf.Depth(mergeds2, 8)
sfdbfks2=fvf.ReplaceFramesSimple(sfdbfks, mergeds2, mappings="[24092 24222]")

sfdbfks2.set_output()