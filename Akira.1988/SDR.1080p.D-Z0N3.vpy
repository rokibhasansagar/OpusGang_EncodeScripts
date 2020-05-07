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
import muvsfunc as muvf
import blazefunc as bzf

src=core.ffms2.Source("")

src=fvf.Depth(src,16)

src=core.std.Crop(src,left=0,right=0,top=24,bottom=24).std.Trim(24,179063)

#different crop w/ dirty lines 
crop=core.std.Crop(src,left=46,right=46,top=20,bottom=20)
crop=core.fb.FillBorders(crop,bottom=1,right=3,top=1,mode="fillmargins").std.Merge(crop,[1,0])
crop=core.edgefixer.ContinuityFixer(crop,top=4,bottom=3,right=0,left=0,radius=4)
crop=core.std.AddBorders(crop,left=46,right=46,top=20,bottom=20)
crop=sgf.FixColumnBrightnessProtect2(crop,48,5)
crop=sgf.FixColumnBrightnessProtect2(crop,1919-46,42)
crop=sgf.FixColumnBrightnessProtect2(crop,1919-49,5)
src=fvf.ReplaceFramesSimple(src,crop,mappings="[173456 179039]")

b16=src

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=16, grainc=8, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Binarize(10000).std.Inflate().std.Maximum().std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
sfdb=fvf.ReplaceFramesSimple(b16, merged, mappings="[163790 163892] [163598 163673] [141964 142059]")

db = b16.f3kdb.Deband(range=20, y=60, cb=60, cr=60, grainy=16, grainc=8, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Maximum().std.Inflate()
merged = core.std.MaskedMerge(db, b16, mask)
sfdb=fvf.ReplaceFramesSimple(sfdb, merged, mappings="[173456 174109]")
#break 174030

final=fvf.Depth(sfdb,8)
final.set_output()