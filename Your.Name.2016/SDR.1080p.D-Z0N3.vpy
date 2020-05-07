import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import mvsfunc as mvf
import muvsfunc as muf
import kagefunc as kgf
import havsfunc as hvf
import vsTAAmbk as taa
import fag3kdb
import nnedi3_resample as nnrs
import sgvsfunc as sgf
from uvsfunc import *

src = core.ffms2.Source("")

core.max_cache_size=22000

src=mvf.Depth(src,16)

# Dirty line fixes

# Sky issue.  No point in adjusting further than masked FillBorders.  Matching grain is a waste of time for this shit.
mask=core.std.ShufflePlanes(src,0,vs.GRAY).std.Expr("x 43500 < x 100 / x 100 * ?")
cf=core.fb.FillBorders(src,top=6).std.MaskedMerge(src,mask)
bb=fvf.ReplaceFramesSimple(src,cf,mappings="[17495 17533]")

# Dots in top and bottom left corners.
mask=core.std.ShufflePlanes(src,0,vs.GRAY).std.Binarize(25000).std.Invert().std.Maximum().std.Inflate().std.Invert()
cf=core.fb.FillBorders(bb,left=2).std.MaskedMerge(bb,mask)
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[42479 42497]")

# Four pixel wide dirty line across half the left border that gets darker towards the bottom.
cf=core.edgefixer.ContinuityFixer(bb,left=4,right=0,bottom=0,top=0,radius=20).std.Crop(top=1080-400)
top=core.std.Crop(bb,bottom=400)
cf=core.std.StackVertical([top,cf])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[90727 90816]")

# Different flashback/grain scenes.
cf=sgf.bbmod(bb,cTop=1,cBottom=0,cLeft=2,cRight=2)
cf=cf.edgefixer.ContinuityFixer(left=[2,2,2],right=0,bottom=0,top=0,radius=1000)
fb=cf.fb.FillBorders(left=2,mode="fillmargins")
mask=core.std.ShufflePlanes(bb,2,vs.GRAY).std.Binarize(27000).std.Invert().resize.Bilinear(1920,1080)
cf=core.std.MaskedMerge(cf,fb,mask,planes=[1,2])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[5571 5730] [104581 105166]")

cf=core.edgefixer.ContinuityFixer(bb, left=[2,0,2], top=1, right=[2,0,2], bottom=0,radius=5)
cf=core.fb.FillBorders(bb,left=2,right=2,top=0,bottom=0,mode="fillmargins").std.Merge(clipb=cf,weight=[1,0,0])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[105323 105406] [106427 106474]")
	
cf=core.edgefixer.ContinuityFixer(bb, left=[2,0,0], top=[2,0,0], right=[2,0,0], bottom=[2,0,0],radius=5)
cf=core.fb.FillBorders(bb,left=2,right=0,top=0,bottom=0,mode="fillmargins").std.Merge(clipb=cf,weight=[1,0,0])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[106767 108038] [105529 105610] [106475 106618] [106671 108038] [108503 109904]")

cf=core.edgefixer.ContinuityFixer(bb, left=[2,2,2], top=[2,2,2], right=[2,2,2], bottom=0,radius=5)
cf=core.fb.FillBorders(bb,left=2,right=0,top=0,bottom=0,mode="fillmargins").std.Merge(clipb=cf,weight=[1,0,0])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[105407 105472] [104615 104698] [105611 106426] [106619 106670]")

# Single columns missing left and right.  Chroma missing on right border.
cf=core.fb.FillBorders(bb,left=1,right=1,mode="fillmargins")
cf=core.fb.FillBorders(cf,left=2,mode="fillmargins").std.Merge(cf,[1,0])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[46584 46619] [64834 64869] [37247 37366] [54042 54257] [54318 54437] [71127 71174] [75005 75052] [79985 80128] [112235 112270] [116463 116498] [116555 116644] [117732 117761] [135403 135480] [143575 143586] [143599 143610]")

# One missing and one heavily damaged border on the right.
cf=core.fb.FillBorders(bb,right=1,mode="fillmargins").edgefixer.ContinuityFixer(left=0,right=2,bottom=0,top=0,radius=150)
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[43903 43903]")

# TV and screen fixes.  First two are commented out since they seem to be intentional.
#cf=core.fb.FillBorders(bb,left=7,right=7,top=2,mode="fillmargins")
#cf=core.std.Merge(bb,cf,[0,1]).edgefixer.ContinuityFixer(left=[10,0,0],right=[5,0,0],top=0,bottom=0,radius=10)
#bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[137359 137508]")
#
#cf=core.fb.FillBorders(bb,left=5,right=5,top=4,bottom=6,mode="fillmargins")
#cf=core.std.Merge(bb,cf,[0,1]).edgefixer.ContinuityFixer(left=[5,0,0],right=[5,0,0],top=0,bottom=0,radius=10)
#bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[9122 9229]")

cf=core.edgefixer.ContinuityFixer(bb, left=[0,0,0], top=[1,0,0], right=[1,0,0], bottom=[1,0,0],radius=5)
cf=core.fb.FillBorders(bb,left=2,right=0,top=2,bottom=2,mode="fillmargins").std.Merge(clipb=cf,weight=[1,0,0])
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[44624 44696]")

# More complicated TV fix.  Bottom is partially damaged and hard to mask around.
fb=core.fb.FillBorders(bb,bottom=2)
luma=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Crop(top=1079)
top=core.std.BlankClip(bb,height=1079,width=1920,format=vs.GRAY16)
layer_a=luma.std.Invert().std.Binarize(40000)
layer_b=core.std.Expr([layer_a,luma.std.Invert().std.Binarize(30000)],"y x -")
layer_c=luma.std.Binarize(40000)
mask_a=core.std.StackVertical([top,layer_a])
mask_b=core.std.StackVertical([top,layer_b])
mask_c=core.std.StackVertical([top,layer_c])
cf_a=sgf.FixRowBrightnessProtect2(bb,1079,50)
cf_b=fb#sgf.FixRowBrightnessProtect2(bb,1079,50)
cf_c=core.edgefixer.ContinuityFixer(bb,left=0,top=0,right=0,bottom=3,radius=15)
merge_a=core.std.MaskedMerge(bb,cf_a,mask_a)
merge_b=core.std.MaskedMerge(merge_a,cf_b,mask_b)
merge_c=core.std.MaskedMerge(merge_b,cf_c,mask_c)
cf=merge_c.edgefixer.ContinuityFixer(left=0,top=0,right=0,bottom=2,radius=15)
cf=core.std.Merge(cf,fb,[0,1])
mask=core.std.StackVertical([top,luma.std.Binarize(24500)])
cf=core.std.MaskedMerge(cf,fb,mask)
mask=core.resize.Spline36(core.std.ShufflePlanes(cf,1,vs.GRAY),1920,1080).std.Binarize(34000)
cf=core.std.MaskedMerge(cf,fb,mask)
bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[127071 127118]")

# Anti-aliasing

# Masked in order to keep straight lines from being affected.
mask=kgf.retinex_edgemask(bb).std.Maximum(coordinates=[0,1,0,0,0,0,1,0]).std.Binarize(65500).std.Minimum(coordinates=[1,0,1,1,1,1,0,1])
aa=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.ReplaceFramesSimple(bb,aa,mappings="[1788 1823] [10139 10203] [12124 12195] [12256 12327] [13732 13827] [13828 13911] [20510 20543] [21066 21113] [26303 26350] [26459 26542] [27131 27166] [35935 35974] [39163 39198] [43478 43519] [44036 44107] [44256 44327] [56586 56657] [59730 59909] [60102 60365] [63388 63495] [74369 74404] [96050 96189] [96562 96717] [97588 97983] [123642 123737] [127119 127166] [127203 127250] [130766 130819] [134053 134112] [134113 134172] [138271 138330] [143599 143610] [19518 19697]")

mmask=core.std.ShufflePlanes(src,0,vs.GRAY).motionmask.MotionMask()
mask=core.std.Expr([mask,mmask],"x y -",vs.GRAY).std.Inflate()
aa=taa.TAAmbk(bb,aatype=-3,preaa=0,aarepair=0,mtype=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,"[29137 29254]")

mask=kgf.retinex_edgemask(bb).std.Maximum(coordinates=[0,1,0,1,1,0,1,0]).std.Binarize(65500).std.Minimum(coordinates=[1,0,1,1,1,1,0,1])
aa=rektaa(bb,left=1380,top=0,right=2,bottom=230,opencl=True,opencl_device=2,mtype=0,strength=0.3)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[9230 9301] ")

mask=kgf.retinex_edgemask(bb).std.Maximum(coordinates=[0,0,0,0,0,1,0,1]).std.Binarize(65500).std.Minimum()
aa=rektaa(bb,left=2,top=286,right=4,bottom=158,opencl=True,opencl_device=2,mtype=0,strength=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[10660 10755]")

# Mask to emphasize diagonal lines.
mask=kgf.retinex_edgemask(bb).std.Binarize(65530).std.Maximum().std.Inflate().std.Minimum(coordinates=[1,0,1,1,1,1,0,1])
aa=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=0)
aa=core.std.MaskedMerge(bb,aa,mask)
aa=rekt(aa,bb,left=614,top=156,right=812,bottom=0)
aa=rekt(aa,bb,left=614,top=628,right=652,bottom=0)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[10288 10359]")

mask=kgf.retinex_edgemask(bb).std.Binarize(65500).std.Inflate().std.Minimum(coordinates=[0,1,0,1,1,0,1,0])
aa=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[16430 16705]")

aa=rektaa(bb,288,350,1600,652)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[10921 11079]")

# Masking white area.
mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(33000).std.Inflate().std.Maximum()
aa1=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=0)
aa=core.std.MaskedMerge(bb,aa1,mask)
aa=rektaa(aa,912,0,968,830)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[16334 16369]")

mask=kgf.retinex_edgemask(bb).std.Binarize(65300)
aa=core.std.MaskedMerge(bb,aa1,mask)
sf=fvf.rfs(sf,aa,mappings="[26543 26614] [26675 26818] [27203 27286] [27287 27316] [33685 33780] [33973 34080] [34297 34380] [35815 35934] [36443 36526] [36734 36838] [41057 41128] [47718 47825] [90887 90922] [92168 92203] [99526 99669] [133723 133818] [136327 136446] [103295 103369]")

rev=rekt(sf,bb,left=342,top=0,right=630,bottom=0)
sf=fvf.rfs(sf,rev,mappings="[99526 99669]")

rev=rekt(aa1,bb,left=0,top=326,right=1532,bottom=354)
rev=rekt(rev,bb,left=1326,top=188,right=400,bottom=466)
sf=fvf.rfs(sf,rev,mappings="[98902 98997]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(30500).std.Inflate().std.Maximum().std.Inflate()
aa=taa.TAAmbk(bb,aatype=6,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[20616 20687]")

aa=rektaa(bb,200,422,1302,468)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[16706 16825]")

luma=core.std.ShufflePlanes(sf,0,vs.GRAY)
mask=rekt_fast(luma,lambda x: core.std.Binarize(x,30000),left=642,top=302,right=768,bottom=500).std.Binarize(65300)
aa=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=2)
maa=core.std.MaskedMerge(sf,aa,mask)
mask=kgf.squaremask(bb,width=1920-(776+1096),height=1080-(970+96),offset_x=1096,offset_y=970)
aa=core.std.MaskedMerge(maa,aa,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[12040 12123]")

aa=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=2)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[20784 20915] [27011 27082] [60498 60593]")

# Mitsuha in room on floor
aa=taa.TAAmbk(bb,aatype=3,preaa=0,strength=0,opencl=True,opencl_device=2,mtype=1)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[6911 7078]")

# Cropping instead of masking properly.
aa=rektaa(bb,left=1126,top=8,right=444,bottom=1016,strength=0)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[8715 8733]")

aa=rektaa(bb,left=1600,top=684,right=2,bottom=26,strength=0,opencl=True,opencl_device=2)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[9302 9349]")

aa=rektaa(bb,left=1560,top=690,right=308,bottom=96,strength=0,opencl=True,opencl_device=2)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[9398 9529]")

aa=rektaa(bb,124,88,80,908)
mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(19500).std.Inflate()
aa1=rektaa(aa,92,702,1756,140,mtype=0,strength=0)
aa=core.std.MaskedMerge(aa,aa1,mask)
aa1=rektaa(aa,1736,702,112,286,mtype=0,strength=0)
aa=core.std.MaskedMerge(aa,aa1,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[20916 21065]")

aa=rektaa(sf,396,820,1514,206)
aa=core.std.MaskedMerge(sf,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[22728 22799]")

aa=rektaa(bb,156,4,604,920)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[23245 23325]")

aa=rektaa(bb,112,420,1412,320)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[23398 23493]")

mask=core.std.ShufflePlanes(bb,2,vs.GRAY).std.Binarize(30500).std.Invert().resize.Spline36(1920,1080).std.Inflate()
aa=rektaa(bb,134,300,442,6)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.ReplaceFramesSimple(sf,aa,mappings="[23758 23877]")

aa=rektaa(bb,544,60,546,952,strength=0)
sf=fvf.rfs(sf,aa,"[24214 24297]")

mask=kgf.retinex_edgemask(bb).std.Binarize(65500)
aa=rektaa(bb,18,70,1700,940,aatype=3,mtype=0,strength=0.5)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[27083 27130]")

aa=rektaa(bb,670,134,1212,930)
sf=fvf.rfs(sf,aa,mappings="[27851 27922]")

aa=rektaa(bb,436,226,1420,724)
sf=fvf.rfs(sf,aa,mappings="[28319 28358]")

aa=rektaa(bb,286,284,1560,706)
sf=fvf.rfs(sf,aa,mappings="[30635 30694] [31585 31740] [39267 39374]")

aa=taa.TAAmbk(bb,3,mtype=1,opencl=True,opencl_device=2).std.Crop(bottom=704)
clip=core.std.Crop(bb,top=1080-704)
aa=core.std.StackVertical([aa,clip])
sf=fvf.rfs(sf,aa,mappings="[31849 32040] [32668 32712]")

aa=rektaa(bb,516,702,516,288)
sf=fvf.rfs(sf,aa,mappings="[32257 32328] [32449 32556] [45664 45705]")

aa=rektaa(bb,982,194,862,862)
sf=fvf.rfs(sf,aa,mappings="[33457 33517]")

aa=rektaa(bb,1016,518,742,174,strength=0)
sf=fvf.rfs(sf,aa,mappings=" [37919 37990]")

aa=rektaa(bb,1018,66,828,778)
sf=fvf.rfs(sf,aa,mappings="[38422 38466]")

aa=rektaa(bb,592,748,1092,140)
sf=fvf.rfs(sf,aa,mappings="[38467 38502]")

aa=rektaa(bb,444,748,1260,140)
sf=fvf.rfs(sf,aa,mappings="[38623 38730]")

aa=rektaa(bb,866,0,884,916)
sf=fvf.rfs(sf,aa,mappings="[42394 42441]")

aa=rektaa(bb,40,810,1722,218)
sf=fvf.rfs(sf,aa,mappings="[42522 42593] [42808 42867] [44980 45015]")

aa=rektaa(bb,0,0,246,1022)
sf=fvf.rfs(sf,aa,mappings="[43964 44035] [54870 54929] [91868 92167]")

aa=rektaa(bb,1666,0,0,1032)
sf=fvf.rfs(sf,aa,mappings="[44212 44255]")

aa=rektaa(bb,1068,90,706,750)
sf=fvf.rfs(sf,aa,mappings="[46383 46448]")

aa=rektaa(bb,340,0,338,970,strength=0)
sf=fvf.rfs(sf,aa,mappings="[56382 56453]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(20000).std.Maximum().std.Inflate()
aa=taa.TAAmbk(bb,3,strength=0,mtype=2,opencl=True,opencl_device=2)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[61014 61085]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(37000).std.Inflate()
aa=taa.TAAmbk(bb,aatype=6,mtype=2)
aa=rekt(bb,aa,left=616,top=304,right=0,bottom=436)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[61194 61305]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(48000).std.Inflate()
aa=rektaa(bb,900,0,0,884,strength=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[69174 69221]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(24000).std.Inflate()
aa=rektaa(bb,0,690,1020,0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[64971 64985]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(30000).std.Maximum().std.Inflate().std.Maximum().std.Inflate()
aa=rektaa(bb,654,284,166,132,strength=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[74957 75004]")

aa=rektaa(bb,676,476,1170,508)
sf=fvf.rfs(sf,aa,mappings="[95842 95901]")

aa=rektaa(bb,0,302,1730,578)
sf=fvf.rfs(sf,aa,mappings="[96190 96345]")

aa=rektaa(bb,1796,0,8,324)
sf=fvf.rfs(sf,aa,mappings="[96514 96561]")

aa=rektaa(bb,1802,674,36,48)
aa=rektaa(aa,1306,26,0,1008,strength=0)
sf=fvf.rfs(sf,aa,mappings="[99322 99357] [99670 99693]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(50000).std.Maximum().std.Inflate()
aa=rektaa(bb,764,66,406,368,strength=0)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,mappings="[99900 99971] [100050 100181]")

aa=rektaa(bb,0,706,1486,258)
sf=fvf.rfs(sf,aa,mappings="[100392 100451]")

aa=rektaa(bb,2,0,0,546)
sf=fvf.rfs(sf,aa,mappings="[119723 119797]")

aa=rektaa(bb,1190,470,470,378,strength=0)
sf=fvf.rfs(sf,aa,mappings="[121470 121565]")

aa=rektaa(bb,0,118,366,172)
sf=fvf.rfs(sf,aa,mappings="[125534 125605]")

aa=rektaa(bb,738,334,1062,356)
sf=fvf.rfs(sf,aa,mappings="[126449 126484]")

aa=rektaa(bb,200,254,882,620,mtype=2,preaa=2,strength=0.3)
sf=fvf.rfs(sf,aa,mappings="[126951 126998]")

aa=rektaa(bb,1466,102,258,108)
sf=fvf.rfs(sf,aa,mappings="[133639 133722]")

aa=taa.TAAmbk(bb,aatype=3,mtype=2,strength=0.3)
aa=rekt(aa,bb,left=382,top=350,right=376,bottom=140)
sf=fvf.rfs(sf,aa,mappings="[137185 137262]")

aa=rektaa(bb,114,420,1414,0)
sf=fvf.rfs(sf,aa,mappings="[122394 122525]")

# stupid sign
mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(59500).std.Maximum()
aa=rektaa(bb,1568,130,224,154,preaa=-1,strength=0)
aa=core.std.MaskedMerge(bb,aa,mask)
aa=rekt(aa,bb,left=1598,top=160,right=238,bottom=180)
sf=fvf.rfs(sf,aa,mappings="[24022 24058]")

aa=rektaa(bb,332,162,1126,826,mtype=3,strength=0,preaa=0,aatype=6,aatypeu=0,aatypev=0,postaa=1)
sf=fvf.rfs(sf,aa,"[1704 1751]")

aa=rektaa(bb,1428,80,0,320,strength=0,preaa=0,aatype=3,postaa=1)
sf=fvf.rfs(sf,aa,"[1860 1895]")

aa=rektaa(bb,886,68,870,500,strength=0,preaa=0,aatype=3,postaa=0)
sf=fvf.rfs(sf,aa,"[5575 5592] [104587 104614] [107643 107678] [109555 109570] [134027 134052]")

aa=rektaa(bb,786,14,628,882,strength=0,preaa=0,aatype=3,postaa=0)
sf=fvf.rfs(sf,aa,"[5863 6090]")

mask=core.std.ShufflePlanes(bb,2,vs.GRAY).std.Binarize(39000).std.Inflate().std.Maximum().resize.Bilinear(1920,1080)
aa=taa.TAAmbk(bb,aatype=-2,mtype=0,strength=0,preaa=1,postaa=False,sharp=-0.5,opencl=True,opencl_device=2)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,"[21114 21173] [21386 21479]")

aa=rektaa(bb,486,552,1088,410,aatypeu=0,aatypev=0,aatype=3,preaa=0,strength=0)
sf=fvf.rfs(sf,aa,"[74045 74155]")

aa=rektaa(bb,48,350,988,608,mtype=1)
sf=fvf.rfs(sf,aa,"[51867 51986]")

mask=core.std.ShufflePlanes(bb,1,vs.GRAY).std.Binarize(115).std.Maximum().resize.Bilinear(1920,1080).std.Inflate()
aa=taa.TAAmbk(bb,aatype=3,aatypeu=0,aatypev=0,mtype=1,strength=0,preaa=0,sharp=0.2,aarepair=0,toon=0,postaa=False,opencl=True,opencl_device=2)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,"[48822 49073] [50604 50687]")

aa=rektaa(bb,658,872,454,30,mtype=1,aatype=3,strength=0,preaa=0,postaa=True)
sf=fvf.rfs(sf,aa,"[18682 18797]")

aa=rektaa(bb,376,254,266,340,mtype=1,aatype=3,preaa=2)
sf=fvf.rfs(sf,aa,"[106923 106970]")

left=core.std.Crop(bb,right=960).edgefixer.ContinuityFixer(top=0,bottom=0,left=0,right=[2,2,2],radius=15)
right=core.std.Crop(bb,left=960).edgefixer.ContinuityFixer(top=0,bottom=0,right=0,left=[1,1,1],radius=15)
cf=core.std.StackHorizontal([left,right])
sf=fvf.rfs(sf,cf,"[42938 42965]")

mask=core.std.ShufflePlanes(bb,0,vs.GRAY).std.Binarize(35000).std.Inflate().std.Inflate()
aa=rektaa(bb,1564,500,240,0,mtype=1)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,"[66918 66989]")

mask=kgf.retinex_edgemask(bb).std.Binarize(65000)
aa=taa.TAAmbk(bb,aatype=-3,sharp=0.2)
aa=hvf.FineDehalo(aa,rx=1,ry=1)
aa=core.std.MaskedMerge(bb,aa,mask)
sf=fvf.rfs(sf,aa,"[75362 75448]")

clip=core.resize.Point(sf,matrix_in_s="709",format=vs.RGBS)
cubed=core.timecube.Cube(clip,"")
sf=core.resize.Point(cubed,matrix=1,format=vs.YUV420P16)

uhd=core.ffms2.Source("").edgefixer.ContinuityFixer(2,2,2,2,20)
uhd=core.std.AssumeFPS(uhd,fpsnum=24).std.Trim(46,213)
uhd=fvf.Depth(uhd,16)
uhd=core.dghdrtosdr.DGHDRtoSDR(uhd,fulldepth=True,sat=1.15,tm=0.4,light=135,hue=5,roll=0.6).resize.Spline36(1920,1080, format=vs.YUV444P16)
clip=core.resize.Point(uhd,matrix_in_s="709",format=vs.RGBS)
cubed=core.timecube.Cube(clip,"")
uhd=core.resize.Point(cubed,matrix=1,format=vs.YUV420P16)
sf=sf.std.Trim(0,48653)+uhd+sf.std.Trim(48822,153317)

jpn=core.ffms2.Source("")
jpn=core.std.AssumeFPS(jpn,fpsnum=24)
jpn=sgf.DelFrameProp(jpn)
jpn=fvf.Depth(jpn,16)
sf=fvf.rfs(sf,jpn,"[2412 2813] [62555 62841]")

# Debanding

db=sf.f3kdb.Deband(range=15,y=48,cb=52,cr=52,grainy=10,grainc=15,output_depth=16)
mask=kgf.retinex_edgemask(sf).std.Maximum().std.Inflate().std.Maximum().std.Inflate()
merged=core.std.MaskedMerge(db,sf,mask)
sf=fvf.ReplaceFramesSimple(sf, merged, mappings="[480 516] [38915 38930] [132848 132871]")

db=core.f3kdb.Deband(sf,range=30,y=0,cb=0,cr=54,grainy=5,grainc=32,output_depth=16)
merged=core.std.MaskedMerge(db,sf,mask)
sf=fvf.rfs(sf,merged,"[128483 128572]")

db=sf.f3kdb.Deband(range=15,y=52,cb=16,cr=16,grainy=10,grainc=5,output_depth=16)
mask=kgf.retinex_edgemask(sf).std.Maximum().std.Inflate().std.Maximum().std.Inflate()
merged=core.std.MaskedMerge(db,sf,mask)
sf=fvf.ReplaceFramesSimple(sf, merged, mappings="[291 450] [1056 1247] [3630 3634] [9350 9397] [9578 9592] [46044 46178] [75701 75988] [99862 99875] [103862 103866] [131181 131312]")
# [118035 118998]
db=fvf.GradFun3(sf,thr=0.5,radius=15)
sf=fvf.rfs(sf,db,"[130892 130951]")

mask=core.std.ShufflePlanes(sf,0,vs.GRAY).std.Binarize(15000).std.Inflate().std.Maximum().std.Inflate()
black=core.std.BlankClip(sf,1920,1080,format=vs.YUV420P16)
merge=core.std.MaskedMerge(black,sf,mask)
sf=fvf.rfs(sf,merge," [0 26] [265 289] [451 480] [1248 1367] [2184 2255] [2679 2867] [4810 4944] [24924 24994] [38931 39030] [46999 47117] [63232 63387] [91525 91643] [132872 132991] [133448 133590] [141890 141959] [145638 153317]")

mask=core.std.ShufflePlanes(sf,0,vs.GRAY).std.Binarize(4500).std.Inflate()
black=core.std.BlankClip(sf,1920,1080,format=vs.YUV420P16)
merge=core.std.MaskedMerge(black,sf,mask)
sf=fvf.rfs(sf,merge,"[290 450] [53790 53873]")

final=mvf.Depth(sf,8)

final.set_output()