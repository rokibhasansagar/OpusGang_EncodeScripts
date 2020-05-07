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

src=core.std.Crop(src,left=0,right=0,top=0,bottom=0)

ds=fvf.Debilinear(src,1920,1080)#,yuv444=True)

noresize=core.resize.Spline36(src,1920,1080).std.Crop(top=22,bottom=22).fb.FillBorders(left=2,right=3,top=1,bottom=1,mode="fillmargins").std.AddBorders(top=22,bottom=22)#.std.Merge(src,[0,1])
ds=fvf.ReplaceFramesSimple(ds,noresize,mappings="[0,647]")

ds=fvf.Depth(ds,16)

db=core.f3kdb.Deband(ds,y=30,cb=30,cr=30,grainy=16,grainc=5,dynamic_grain=True,output_depth=16)
mask=kgf.retinex_edgemask(ds).std.Inflate()
merge=core.std.MaskedMerge(db,ds,mask)
ds=fvf.ReplaceFramesSimple(ds,merge,mappings="[648,702] [44473 44594]")

final=fvf.Depth(ds,10)

final.set_output()