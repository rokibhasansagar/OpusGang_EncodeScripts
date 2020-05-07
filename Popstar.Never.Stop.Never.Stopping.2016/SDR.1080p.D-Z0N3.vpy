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

src=core.std.Crop(src,left=0,right=0,top=140,bottom=140)

bb=core.edgefixer.ContinuityFixer(src,left=5,top=0,right=5,bottom=0,radius=10)
bb=fvf.ReplaceFramesSimple(src,bb,mappings="[35 596]")

crop1=core.std.Crop(bb,left=354,right=356)
fbblack=core.fb.FillBorders(crop1,left=1,right=0,top=0,bottom=0,mode="fillmargins")
fbblack=sgf.FixColumnBrightnessProtect2(fbblack,1205,-4)
bbblack=core.edgefixer.ContinuityFixer(fbblack,left=8,top=0,right=3,bottom=0,radius=10)
fb1black=core.fb.FillBorders(crop1,left=3,right=3,top=0,bottom=0,mode="fillmargins")
mergeblack=core.std.Merge(bbblack,fb1black,[0,1])
black1=sgf.BlackBorders(mergeblack,left=354,right=356)
sfcrop1=fvf.ReplaceFramesSimple(bb,black1,mappings="[909 1408] [1540 1745] [1930 2314] [2600 2820] [2950 3217] [8970 9207] [63843 63887] [63926 63959] [64006 64044]")

crop2=core.std.Crop(bb,left=210,right=208)
fbblack=core.fb.FillBorders(crop2,left=1,right=1,top=0,bottom=0,mode="fillmargins")

bbblack=core.edgefixer.ContinuityFixer(fbblack,left=17,top=0,right=17,bottom=0,radius=20)
fb1black=core.fb.FillBorders(bbblack,left=3,right=3,top=0,bottom=0,mode="fillmargins")
mergeblack=core.std.Merge(bbblack,fb1black,[0,1])
black2=sgf.BlackBorders(mergeblack,left=210,right=208)
sfcrop2=fvf.ReplaceFramesSimple(sfcrop1,black2,mappings="[2379 2444]")

crop3=core.std.Crop(sfcrop2,left=278,right=278)
fb3black=core.fb.FillBorders(crop3,left=1,right=0,top=0,bottom=0,mode="fillmargins")
bb3black=core.edgefixer.ContinuityFixer(fb3black,left=2,top=0,right=3,bottom=0,radius=10)
fb3black=core.fb.FillBorders(crop3,left=3,right=3,top=0,bottom=0,mode="fillmargins")
merge3black=core.std.Merge(bb3black,fb3black,[0,1])
black3=sgf.BlackBorders(merge3black,left=278,right=278)
sfcrop3=fvf.ReplaceFramesSimple(sfcrop2,black3,mappings="[3618 4428]")

crop4=core.std.Crop(sfcrop3,left=970,right=494)
bb4black=core.edgefixer.ContinuityFixer(crop4,left=10,top=0,right=5,bottom=0,radius=10)
fb4black=core.fb.FillBorders(bb4black,left=0,right=2,top=0,bottom=0,mode="fillmargins")
merge4black=core.std.Merge(bb4black,fb4black,[0,1])
black4=sgf.BlackBorders(merge4black,left=970,right=0)
black4=core.std.AddBorders(black4,right=494)
sfcrop4=fvf.ReplaceFramesSimple(sfcrop3,black4,mappings="[4921 5048]")

crop5=core.std.Crop(sfcrop4,left=704,right=750)
bb5black=core.edgefixer.ContinuityFixer(crop5,left=4,top=0,right=6,bottom=0,radius=10)
fb5black=core.fb.FillBorders(bb5black,left=0,right=3,top=0,bottom=0,mode="fillmargins")
merge5black=core.std.Merge(bb5black,fb5black,[0,1])
black5=sgf.BlackBorders(merge5black,left=704,right=750)
black5=core.std.AddBorders(black5,right=0)
sfcrop5=fvf.ReplaceFramesSimple(sfcrop4,black5,mappings="[5049 5161]")

crop6=core.std.Crop(sfcrop5,left=230,right=230)
bb6black=core.edgefixer.ContinuityFixer(crop6,left=6,top=0,right=7,bottom=0,radius=10)
fb6black=core.fb.FillBorders(bb6black,left=0,right=2,top=0,bottom=0,mode="fillmargins")
merge6black=core.std.Merge(bb6black,fb6black,[0,1])
black6=sgf.BlackBorders(merge6black,left=230,right=230)
black6=core.std.AddBorders(black6,right=0)
sfcrop6=fvf.ReplaceFramesSimple(sfcrop5,black6,mappings="[5263 5418]")

crop7=core.std.Crop(sfcrop6,left=226,right=226)
bb7black=core.edgefixer.ContinuityFixer(crop7,left=6,top=0,right=6,bottom=0,radius=10)
fb7black=core.fb.FillBorders(bb7black,left=2,right=2,top=0,bottom=0,mode="fillmargins")
merge7black=core.std.Merge(bb7black,fb7black,[0,1]).fb.FillBorders(left=1,right=1,top=0,bottom=0,mode="fillmargins")
black7=sgf.BlackBorders(merge7black,left=226,right=226)
black7=core.std.AddBorders(black7,right=0)
sfcrop7=fvf.ReplaceFramesSimple(sfcrop6,black7,mappings="[13736 14003]")

crop8=core.std.Crop(sfcrop7,left=356,right=356)
bb8black=core.edgefixer.ContinuityFixer(crop8,left=7,right=11,top=0,bottom=0,radius=15)
fb8black=core.fb.FillBorders(bb8black,left=2,right=3,top=0,bottom=0,mode="fillmargins")
merge8black=core.std.Merge(bb8black,fb8black,[0,1])
black8=sgf.BlackBorders(merge8black,left=356,right=356)
sfcrop8=fvf.ReplaceFramesSimple(sfcrop7,black8,mappings="[22741 22897]")

crop9=core.std.Crop(sfcrop8,left=240,right=240)
bb9black=core.edgefixer.ContinuityFixer(crop9,left=5,top=0,right=4,bottom=0,radius=10)
fb9black=core.fb.FillBorders(bb9black,left=0,right=2,top=0,bottom=0,mode="fillmargins")
merge9black=core.std.Merge(bb9black,fb9black,[0,1]).fb.FillBorders(left=0,right=0,top=0,bottom=0,mode="fillmargins")
black9=sgf.BlackBorders(merge9black,left=240,right=240)
sfcrop9=fvf.ReplaceFramesSimple(sfcrop8,black9,mappings="[28671 28754] [28831 28941] [60714 60765] [85303 85536]")

crop10=core.std.Crop(sfcrop9,left=656,right=656)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=4,top=0,right=4,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=0,right=2,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=1,right=1,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=656,right=656)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop9,black10,mappings="[28755 28830] [28942 28975]")

crop10=core.std.Crop(sfcrop10,left=258,right=258)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=6,top=0,right=5,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=0,right=0,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=0,right=0,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=258,right=258)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[50890 51137]")

crop10=core.std.Crop(sfcrop10,left=686,right=688)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=6,top=0,right=5,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=2,right=2,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=0,right=0,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=686,right=688)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[52724 52790] [52994 53031]")

crop10=core.std.Crop(sfcrop10,left=210,right=210)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=6,top=0,right=5,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=2,right=2,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=0,right=0,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=210,right=210)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[58083 58222]")

crop10=core.std.Crop(sfcrop10,left=424,right=424)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=5,top=0,right=7,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=0,right=0,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=0,right=0,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=424,right=424)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[66009 66070]")

crop10=core.std.Crop(sfcrop10,left=232,right=234)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=4,top=0,right=4,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=2,right=4,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=1,right=1,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=232,right=234)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[66071 66239]")

crop10=core.std.Crop(sfcrop10,left=600,right=616)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=6,top=0,right=5,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=3,right=0,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=1,right=0,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=600,right=616)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[78671 78721]")

crop10=core.std.Crop(sfcrop10,left=160,right=160)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=5,top=0,right=5,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=2,right=2,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=1,right=1,top=0,bottom=0,mode="fillmargins")
black10=sgf.BlackBorders(merge10black,left=160,right=160)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[84172 84184]")

crop10=core.std.Crop(sfcrop10,left=160,right=160,bottom=134)
crop11=core.std.Crop(sfcrop10,left=160,right=160,top=666)
bb10black=core.edgefixer.ContinuityFixer(crop10,left=7,top=0,right=5,bottom=0,radius=10)
fb10black=core.fb.FillBorders(bb10black,left=2,right=2,top=0,bottom=0,mode="fillmargins")
merge10black=core.std.Merge(bb10black,fb10black,[0,1]).fb.FillBorders(left=1,right=1,top=0,bottom=0,mode="fillmargins")
bb11black=core.edgefixer.ContinuityFixer(crop11,left=3,right=0,top=0,bottom=10,radius=5)
merge10black=core.std.StackVertical([merge10black,bb11black])
black10=sgf.BlackBorders(merge10black,left=160,right=160)
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,black10,mappings="[84185 84279]")

left=core.std.Crop(sfcrop10,right=1920-312).edgefixer.ContinuityFixer(left=0,right=[5,5,5],top=0,bottom=0,radius=15)
mid=core.std.Crop(sfcrop10,left=312,right=312).edgefixer.ContinuityFixer(left=5,right=6,top=0,bottom=0,radius=10)
fb=core.fb.FillBorders(mid,right=4,left=6)
mid=core.std.Merge(mid,fb,[0,1])
right=core.std.Crop(sfcrop10,left=1920-312).edgefixer.ContinuityFixer(left=5,right=0,top=0,bottom=0,radius=10)
merge=core.std.StackHorizontal([left,mid,right])
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,merge,mappings="[3360 3401]")

left=core.std.Crop(sfcrop10,right=1920-554).edgefixer.ContinuityFixer(left=0,right=[5,5,5],top=0,bottom=0,radius=15)
mid=core.std.Crop(sfcrop10,left=554,right=554).edgefixer.ContinuityFixer(left=5,right=6,top=0,bottom=0,radius=10)
fb=core.fb.FillBorders(mid,right=4,left=6)
mid=core.std.Merge(mid,fb,[0,1])
right=core.std.Crop(sfcrop10,left=1920-554).edgefixer.ContinuityFixer(left=5,right=0,top=0,bottom=0,radius=10)
merge=core.std.StackHorizontal([left,mid,right])
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,merge,mappings="[4623 4673]")

left=core.std.Crop(sfcrop10,right=1920-336).edgefixer.ContinuityFixer(left=0,right=[5,5,5],top=0,bottom=0,radius=15)
mid=core.std.Crop(sfcrop10,left=336,right=336).edgefixer.ContinuityFixer(left=5,right=6,top=0,bottom=0,radius=10)
fb=core.fb.FillBorders(mid,right=6,left=6)
mid=core.std.Merge(mid,fb,[0,1])
right=core.std.Crop(sfcrop10,left=1920-336).edgefixer.ContinuityFixer(left=5,right=0,top=0,bottom=0,radius=10)
merge=core.std.StackHorizontal([left,mid,right])
sfcrop10=fvf.ReplaceFramesSimple(sfcrop10,merge,mappings="[33891 33968]")

splitmidcropleft=core.std.Crop(sfcrop10,left=964).edgefixer.ContinuityFixer(left=4,top=0,right=0,bottom=0,radius=10)
splitmidcropright=core.std.Crop(sfcrop10,right=958).edgefixer.ContinuityFixer(left=0,top=0,right=6,bottom=0,radius=10).std.AddBorders(right=2)
splitmid=core.std.StackHorizontal([splitmidcropright,splitmidcropleft])
sfsplitmid=fvf.ReplaceFramesSimple(sfcrop10,splitmid,mappings="[14754 14781]")

splitmidcropleft=core.std.Crop(sfcrop10,left=970).edgefixer.ContinuityFixer(left=2,top=0,right=0,bottom=0,radius=10)
splitmidcropleft=core.fb.FillBorders(splitmidcropleft,left=2,right=0,top=0,bottom=0,mode='fillmargins').std.Merge(clipb=splitmidcropleft,weight=[1,0])
splitmidcropright=core.std.Crop(sfcrop10,right=960).edgefixer.ContinuityFixer(left=0,top=0,right=[9,1,1],bottom=0,radius=15)
splitmidcropright=core.fb.FillBorders(splitmidcropright,left=0,right=1,top=0,bottom=0,mode='fillmargins').std.AddBorders(right=10)#.std.Merge(clipb=splitmidcropright,weight=[1,0])
splitmid=core.std.StackHorizontal([splitmidcropright,splitmidcropleft])
sfsplitmid=fvf.ReplaceFramesSimple(sfcrop10,splitmid,mappings="[9756 9787]")

endcredtop=core.std.Crop(sfsplitmid,bottom=800-108)
endcredmid=core.std.Crop(sfsplitmid,top=108,bottom=108).edgefixer.ContinuityFixer(left=0,top=4,right=0,bottom=4,radius=10)
endcredbot=core.std.Crop(sfsplitmid,top=800-108)
splitmid=core.std.StackVertical([endcredtop,endcredmid,endcredbot])
endcredleft=core.std.Crop(sfsplitmid,right=1920-30)
endcredmid=core.std.Crop(splitmid,left=30,right=854).edgefixer.ContinuityFixer(left=4,top=0,right=4,bottom=0,radius=10).fb.FillBorders(left=0,right=1,top=0,bottom=0,mode='fillmargins')
endcredright=core.std.Crop(sfsplitmid,left=1920-854)
splitmid=core.std.StackHorizontal([endcredleft,endcredmid,endcredright])
sfsplitmid=fvf.ReplaceFramesSimple(sfcrop10,splitmid,mappings="[119754 121204]")

contfix1=core.edgefixer.ContinuityFixer(sfsplitmid,left=7,top=0,right=0,bottom=0,radius=10)
sfcontfix1=fvf.ReplaceFramesSimple(sfsplitmid,contfix1,mappings="[19991 20003] [20215 20271]")

b16=fvf.Depth(sfcontfix1,16)

aa = taa.TAAmbk(b16,aatype=-3,preaa=-1,strength=-0.5,mtype=2)
sfaa=fvf.ReplaceFramesSimple(b16,aa,mappings="")

aa2=taa.TAAmbk(sfaa,aatype=-3,preaa=-1,strength=0.4,mtype=2)
sfaa2=fvf.ReplaceFramesSimple(sfaa,aa2,mappings="")

aa3=taa.TAAmbk(sfaa2,aatype=-3,preaa=-1,strength=0.2,mtype=2)
sfaa3=fvf.ReplaceFramesSimple(sfaa2,aa3,mappings="")

dbfk = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=32, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Maximum().std.Inflate()
merged = core.std.MaskedMerge(dbfk, b16, mask)
sfdbfk=fvf.ReplaceFramesSimple(sfaa3, merged, mappings="[622 652] [702 736] [1883 1929] [3218 3401] [3488 3552] [4429 4464] [5447 5472] [10395 10575] [10646 10719] [13111 13316] [15693 15716] [15874 15889] [19837 19883] [21914 22054] [22310 22386] [25725 25801] [28976 29020] [29021 29122] [31673 31673] [33891 33968] [50890 51137] [56725 56778] [57201 57209] [61545 61952] [65468 65571] [78869 79069] [106636 106705] [118600 118673]")
# TV  [27156 27163]

dbfks = b16.f3kdb.Deband(range=20, y=48, cb=48, cr=48, grainy=32, grainc=10, dynamic_grain=True, output_depth=16)
masks = kgf.retinex_edgemask(b16).std.Maximum().std.Inflate()
mergeds = core.std.MaskedMerge(dbfks, b16, masks)
sfdbfks=fvf.ReplaceFramesSimple(sfdbfk, mergeds, mappings="[1465 1539] [4533 4587] [5162 5262] [5608 5658] [6742 6775] [7131 7178] [14740 14753] [15037 15113] [15166 15227] [15252 15297] [15333 15352] [15384 15424] [16048 16082] [16287 16324] [16325 16428] [19739 19771] [19906 19918] [19968 20048] [20049 20120] [20258 20271] [20337 20409] [20410 20483] [20664 20756] [22237 22309] [22868 22896] [26956 27043] [29123 29205] [30100 30110] [34052 34070] [36854 36890] [39156 39199] [39413 39451] [41742 41845] [43435 43594] [43660 43696] [43747 43800] [47486 47531] [53693 53716] [116371 116402] [116623 116660] [119786 119798] [121196 121204]")
#literally iphone [44728 44781]

final=fvf.Depth(sfdbfks,8)

final.set_output()