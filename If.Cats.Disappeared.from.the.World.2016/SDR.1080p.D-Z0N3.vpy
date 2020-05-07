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

src=core.ffms2.Source("").std.Crop(top=138,bottom=138)

#fix=probably some deinterlacer
#src=fvf.ReplaceFramesSimple(src,fix,mappings="[7604 7665]")

bb=sgf.FixRowBrightness(src,0,16,221,16,255)
bb=sgf.FixRowBrightness(bb,1,16,255,16,237)
bb=sgf.FixRowBrightness(bb,2,16,242,16,255)
bb=sgf.FixRowBrightness(bb,3,16,255,16,253)
bb=sgf.FixRowBrightness(bb,4,16,255,16,253)
bb=sgf.FixRowBrightness(bb,5,16,251,16,255)
bb=sgf.FixRowBrightness(bb,6,16,255,16,253)

bb=sgf.FixRowBrightnessProtect2(bb,803,15)
bb=sgf.FixRowBrightness(bb,802,16,255,16,235)
bb=sgf.FixRowBrightness(bb,801,16,240,16,255)
bb=sgf.FixRowBrightness(bb,798,16,250,16,255)

bb=core.edgefixer.ContinuityFixer(bb,left=0,top=2,right=0,bottom=2,radius=10)

b16 = mvf.Depth(bb, 16)

dbfk = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=32, grainc=32, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Binarize(5000).std.Inflate()
merged = core.std.MaskedMerge(dbfk, b16, mask)
merged = mvf.Depth(merged, 8)
sfdbfk=fvf.ReplaceFramesSimple(bb, merged, mappings="")

aa = taa.TAAmbk(sfdbfk,aatype=-3,preaa=-1,strength=-1.5,mtype=2)
sfaa=fvf.ReplaceFramesSimple(sfdbfk,aa,mappings="[38 273]")

sfaa.set_output()