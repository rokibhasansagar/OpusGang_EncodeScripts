import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import vsTAAmbk as taa
import sgvsfunc as sgf

src=core.ffms2.Source("")

src=core.std.Trim(src,24,32848)

#src=sgf.FixColumnBrightnessProtect2(src,1919,40)
#src=sgf.FixColumnBrightnessProtect2(src,1918,20)
#src=sgf.FixColumnBrightnessProtect2(src,1917,-5)
#src=sgf.FixColumnBrightnessProtect2(src,1916,1)
#src=sgf.FixColumnBrightnessProtect2(src,1915,-1)
#src=sgf.FixColumnBrightnessProtect2(src,1914,-2)
#src=core.fb.FillBorders(src,right=4,mode="fillmargins").std.Merge(src,[1,0]).fb.FillBorders(right=1,mode="fillmargins")
#
#src=sgf.FixColumnBrightnessProtect2(src,0,75)
#src=sgf.FixColumnBrightnessProtect2(src,1,20)
#src=sgf.FixColumnBrightnessProtect2(src,2,-10)
#src=core.fb.FillBorders(src,left=2,mode="fillmargins").std.Merge(src,[1,0]).fb.FillBorders(left=1)
#
#src=sgf.FixRowBrightnessProtect2(src,0,14.5)
#src=sgf.FixRowBrightnessProtect2(src,1,-10)
#
#src=sgf.FixRowBrightnessProtect2(src,1079,15)
#src=sgf.FixRowBrightnessProtect2(src,1078,-10)

src=fvf.Depth(src,16)

src=core.edgefixer.ContinuityFixer(src,left=4,right=6,top=3,bottom=3,radius=3)
src=core.fb.FillBorders(src,right=4,left=4,mode="fillmargins").std.Merge(src,[1,0])

ds=fvf.Debicubic(src,1280,720,b=0.6,c=0.2,yuv444=True)

mds=fvf.DebicubicM(src,1280,720,b=0.6,c=0.2,yuv444=True,thr=5,inflate=30)
src=fvf.ReplaceFramesSimple(ds,mds,mappings="[16161 16232]")

intro=core.ffms2.Source("").std.Trim(24,2181)
intro=fvf.Depth(intro,16)
intro=core.edgefixer.ContinuityFixer(intro,left=4,right=6,top=3,bottom=3,radius=3)
intro=core.fb.FillBorders(intro,right=4,left=4,mode="fillmargins").std.Merge(intro,[1,0])
intro=fvf.Debicubic(intro,1280,720,b=0.6,c=0.2,yuv444=True)

outro=core.ffms2.Source("").std.Trim(609,2181)
outro=fvf.Depth(outro,16)
outro=core.edgefixer.ContinuityFixer(outro,left=4,right=6,top=3,bottom=3,radius=3)
outro=core.fb.FillBorders(outro,right=4,left=4,mode="fillmargins").std.Merge(outro,[1,0])
outro=fvf.Debicubic(outro,1280,720,b=0.6,c=0.2,yuv444=True)

src=intro+core.std.Trim(src,2158,30871)+outro

#hardsub [30753 30861]

b16=fvf.Depth(src,16)

aa = taa.TAAmbk(b16,aatype=-3,preaa=-1,strength=-1.5,mtype=2)
sfaa=fvf.ReplaceFramesSimple(b16,aa,mappings="")

aa=taa.TAAmbk(sfaa,aatype=-3,preaa=-1,strength=-5,mtype=2)
sfaa=fvf.ReplaceFramesSimple(sfaa,aa,mappings="")

db=sfaa.f3kdb.Deband(range=5, y=4, cb=4, cr=4, grainy=0, grainc=0, output_depth=16)
mask=kgf.retinex_edgemask(sfaa).std.Binarize(3000).std.Inflate()#.std.Expr("x 2000 > x 10 * x ?").std.Binarize(5000).std.Inflate()
sfdb=core.std.MaskedMerge(db,sfaa,mask)

db = sfaa.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=5, output_depth=16)
mask = kgf.retinex_edgemask(sfaa).std.Inflate()#.std.Maximum().std.Inflate()#.std.Expr("x 2000 > x 10 * x ?").std.Binarize(5000).std.Inflate()
merged = core.std.MaskedMerge(db, sfaa, mask)
sfdb=fvf.ReplaceFramesSimple(sfdb, merged, mappings="[5382 5439] [8059 8129] [9018 9100] [11212 11620] [26374 26445] [26517 26579]")
#weird door  [19473 19505]

db=sfaa.f3kdb.Deband(range=20,y=180,cb=4,cr=4,grainy=15,grainc=0,output_depth=16)
#test=fvf.Depth(sfaa,8)
mask=sfaa.std.Expr("x 22000 < x 10 / x ?").std.Maximum().std.Inflate().std.Binarize(22000)
merged = core.std.MaskedMerge(sfaa, db, mask)
#merged=sgf.LumaMaskMerge(sfaa,db,128**2,True)
sfdb=fvf.ReplaceFramesSimple(sfdb,merged,mappings="[14185 14219]")

final=fvf.Depth(sfdb,10)

final.set_output()
