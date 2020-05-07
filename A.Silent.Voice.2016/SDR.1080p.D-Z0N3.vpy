import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import mvsfunc as mvf
import kagefunc as kgf
import havsfunc as hvf
import vsTAAmbk as taa
import fag3kdb
import sgvsfunc as sgf
import nnedi3_resample as nnrs
    
src = core.ffms2.Source("")

src = core.std.Crop(src, 0, 0, 22, 22)

dn = mvf.BM3D(src, sigma=1, radius1=1) #denoise
dn=fvf.ReplaceFramesSimple(src,dn,mappings="")
# [4471 4590]

#progress 72427

aa = taa.TAAmbk(dn,aatype=-2,strength=0.5)
sf=fvf.ReplaceFramesSimple(dn,aa,mappings="[]")#[107671 107897] [112407 112588]

bb=core.edgefixer.ContinuityFixer(src, left=0, top=4, right=0, bottom=4, radius=10)
dn=fvf.ReplaceFramesSimple(sf,bb,mappings="[0 359]")

bb1=core.edgefixer.ContinuityFixer(src, left=0, top=4, right=0, bottom=0, radius=10)
dn=fvf.ReplaceFramesSimple(dn,bb1,mappings="[19081 19114]")

bb2=core.edgefixer.ContinuityFixer(src, left=0, top=5, right=0, bottom=0,radius=10)
fb1=core.fb.FillBorders(bb2,left=0,right=0,top=5,bottom=0,mode="fillmargins")
merge1=core.std.Merge(bb2,fb1,[0,1])
dn=fvf.ReplaceFramesSimple(dn,merge1,mappings="[45655 45701] [45776 45858]")

#[45655 45701] [45776 45858] is what Geek thinks is a dirty line

#b16 = mvf.Depth(dn, 16)
#
##worth it
#dbdn=mvf.BM3D(b16,sigma=8,radius1=1)#b16.dfttest.DFTTest(sigma=8,tbsize=1,sbsize=24,sosize=18)
#dbf=dbdn.f3kdb.Deband(range=15,y=64,cb=0,cr=0,grainy=16,grainc=0,output_depth=16)
#mask=kgf.retinex_edgemask(dbdn).std.Binarize(5000).std.Inflate()
#merged=core.std.MaskedMerge(dbf,dbdn,mask)#fag3kdb.Fag3kdb(dbdn,radiusy=15,radiusc=0,thry=88,thrc=0,grainy=32,dynamic_grainy=True)
#dbmake=core.std.MakeDiff(merged,b16,planes=0)
#dbmerge=core.std.MergeDiff(merged,dbmake,planes=0)
#dbf8=mvf.Depth(dbmerge,8)
#sfdbf=fvf.ReplaceFramesSimple(dn, dbf8, mappings="[1148 1277] [32980 33036] [50995 51045] [66689 66805] [76171 76288] [76876 77053] [77954 78141] [78227 78339] [80169 80185] [80329 80348] [80732 80753] [80846 80997] [80998 81030] [81387 81506] [85332 85413] [86600 86675] [91853 92058] [92059 92242] [92278 92324] [92577 92619] [97387 97415] [97416 97467] [97600 97674] [97704 97738] [100466 100574] [100575 100646] [103056 103199] [103506 103546] [105259 105364] [119478 119635] [123648 123805] [128279 128335] [129699 129754] [131435 131454] [131473 131552] [131587 131636] [132028 132130] [132932 133282] [133898 133969] [136293 138519] [138573 139934] [140124 140227] [140362 141120] [141188 141358] [141403 142166] [142285 142400] [142482 142927] [143244 143527] [143973 144186] [144543 144914] [145281 145316] [145557 145714] [145834 146042] [146319 146602] [147539 147576] [147795 147936] [147974 148124] [148253 148349] [148386 148505] [148542 149036] [149171 149244] [149359 149473] [149658 149710] [149711 149829] [150148 150188] [151898 151959] [151990 152037] [154786 154808] [154839 154873] [154924 154971] [155122 155223] [156977 157096] [157144 157279] [157325 157380] [158762 158835] [158863 158926] [159440 159466] [161897 162029] [162169 162242] [162766 163127] [163489 163638] [164476 164563] [165677 165786] [165894 165948] [166853 167013] [167109 167174] [169002 169164] [171044 171110] [171156 171197] [173041 173100] [173193 173474] [173542 173715] [173895 174126] [174195 174301] [175463 175483] [176538 176596] [177690 177785] [180670 180733]")
#
##kind of worth it
#dbdn1=mvf.BM3D(b16,sigma=5,radius1=1)#b16.dfttest.DFTTest(sigma=8,tbsize=1,sbsize=24,sosize=18)
#dbf1=dbdn1.f3kdb.Deband(range=15,y=64,cb=0,cr=0,grainy=16,grainc=0,output_depth=16)
#mask1=kgf.retinex_edgemask(dbdn).std.Binarize(5000).std.Inflate()
#merged1=core.std.MaskedMerge(dbf1,dbdn1,mask)#fag3kdb.Fag3kdb(dbdn,radiusy=15,radiusc=0,thry=88,thrc=0,grainy=32,dynamic_grainy=True)
#dbmake1=core.std.MakeDiff(merged1,b16,planes=0)
#dbmerge1=core.std.MergeDiff(merged1,dbmake1,planes=0)
#dbf81=mvf.Depth(dbmerge1,8)
#sfdbf=fvf.ReplaceFramesSimple(sfdbf,dbf81,mappings="[35408 35541] [40451 40652] [41401 41487] [44144 44223] [45655 45701] [45776 45858] [50525 50589] [68306 68379] [70219 70380] [70961 70998] ")
#
##not worth it [5133 5191] [30054 30136] [35311 35407] [49181 49366] [69346 69435] [69928 69977] [70381 70425] [72210 72427] [72874 73316] [73646 73819] 
#
##new test
#dbf2=b16.f3kdb.Deband(range=15,y=100,cb=0,cr=0,grainy=0,grainc=0,output_depth=16)
#mask2=kgf.retinex_edgemask(b16)#.std.Binarize(5000).std.Inflate()
#merged2=core.std.MaskedMerge(dbf2,b16,mask)#fag3kdb.Fag3kdb(dbdn,radiusy=15,radiusc=0,thry=88,thrc=0,grainy=32,dynamic_grainy=True)
#dbmake2=core.std.MakeDiff(merged2,b16,planes=0)
#dbmerge2=core.std.MergeDiff(merged2,dbmake2,planes=0)
#dbf82=mvf.Depth(dbmerge2,8)
#sfdbf2=fvf.ReplaceFramesSimple(dn,dbf82,mappings="[45655 45701] [141403 142166]")
#
#resize=core.resize.Spline36(dn,1280,688)

dn.set_output()
#resize.set_output()
#sfdbf.set_output()
#sfdbf2.set_output()

#test=core.std.Trim(sfdbf,30054,30136)+core.std.Trim(sfdbf,44144,44223)+core.std.Trim(sfdbf,162766,162866)+core.std.Trim(sfdbf,105259,105364)
#every=sgf.SelectRangeEvery(dn,every=39000,length=250,offset=20000)
#every.set_output()


#wtf [103356 103358] [157097 157117