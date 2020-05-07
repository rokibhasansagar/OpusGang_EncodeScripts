import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
import vsTAAmbk as taa
import fag3kdb
import nnedi3_resample as nnrs
import sgvsfunc as sgf
import nnedi3_rpow2 as nnrp

src = core.ffms2.Source("")
resize = src # I called this and was too lazy to change it.

# Rescaling a flashback with grain.
dn = core.knlm.KNLMeansCL(src, d=3, a=1, s=0, h=1.5, device_type="gpu", device_id=1, channels="Y")
diff = core.std.MakeDiff(src, dn, planes=[0])
ds = fvf.Debicubic(dn, 1280, 720)
us = nnrp.nnedi3_rpow2(ds, 2, 1920, 1080, kernel="Spline36")
merged = core.std.MergeDiff(us, diff, planes=[0])
src = fvf.ReplaceFramesSimple(resize, merged, mappings="[3418 3507] [3508 5145] [75916 76205] [76253 76323] [77720 77790]")

# Rescaling a flashback without grain.
ds = fvf.DescaleAA(dn, 1280, 720).std.MergeDiff(diff, planes=[0])
src = fvf.ReplaceFramesSimple(src, ds, mappings="[3298 3417]")

# Dithering to 16-bit.  The above parts are in 8-bit because I was scared of performance issues.
src = fvf.Depth(src, 16)

# I like to establish a separate variable for 16-bit and leave src for 8-bit, but didn't do that here.  This is so I could copy-paste commands.
b16 = src

# Anti-aliasing.  As you might be able to tell, the crop and stacking could now be replaced by rekt_fast or simply rektaa.
aa = core.std.Crop(b16, left=400, right=1006)
aa = taa.TAAmbk(aa, aatype=-3, preaa=-1, strength=0, mtype=2, opencl=True)
left = core.std.Crop(b16, right=1920 - 400)
right = core.std.Crop(b16, left=1920 - 1006)
aa = core.std.StackHorizontal([left, aa, right]).std.Crop(top=208, bottom=456)
top = core.std.Crop(b16, bottom=1080 - 208)
bottom = core.std.Crop(b16, top=1080 - 456)
aa = core.std.StackVertical([top, aa, bottom])
sfaa = fvf.ReplaceFramesSimple(b16, aa, mappings="[42583 42813] [58812 59050] [65211 65281] [92132 92274]")

# Debanding with a standard ass mask.
db = b16.f3kdb.Deband( range=15, y=60, cb=60, cr=60, grainy=22, grainc=22, output_depth=16)
mask = kgf.retinex_edgemask(b16).std.Inflate() 
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(sfaa, merged, mappings="[3508 3603] [17600 17706] [41865 42113] [76922 77488] [78444 78598] [81054 81280] [150853 150933] [152057 152288] [152324 152424] [152443 152508] [152521 152686] [171669 172433] [172561 172643] [170283 170557]")

# Debanding values that were outside of the range of 10000-25000.
db = b16.f3kdb.Deband(range=10, y=160, cb=0, cr=0, grainy=28, grainc=0, output_depth=16)
mask = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Expr("x 10000 < x 25000 > or x 10 * x 10 / ?")
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = fvf.ReplaceFramesSimple(sfdb, merged, mappings=" [96133 96273]")

# Fixing dirty lines during credits.  Again, rektaa would've been useful back then.
bot = core.std.Crop(sfdb, top=1080 - 330)
middle = core.std.Crop(sfdb, top=318, bottom=330).edgefixer.ContinuityFixer(top=1, bottom=1, left=0, right=0, radius=5)
fb = core.fb.FillBorders(middle, top=2, bottom=2)
middle = core.std.Merge(fb, middle, [1, 0])
top = core.std.Crop(sfdb, bottom=1080 - 318)
merge = core.std.StackVertical([top, middle, bot])
right = core.std.Crop(merge, left=1920 - 134)
middle = core.std.Crop(merge, left=1018, right=134).edgefixer.ContinuityFixer(left=2, right=2, top=0, bottom=0, radius=5)
fb = core.fb.FillBorders(middle, left=2, right=2)
middle = core.std.Merge(fb, middle, [1, 0])
left = core.std.Crop(merge, right=1920 - 1018)
merge = core.std.StackHorizontal([left, middle, right])
sfc = fvf.ReplaceFramesSimple(sfdb, merge, mappings="[165067 167168] [167403 169466] [169842 170557] [170558 171041]")

# Dithering the result back to 8-bit.
final = fvf.Depth(sfc, 8)

final.set_output()