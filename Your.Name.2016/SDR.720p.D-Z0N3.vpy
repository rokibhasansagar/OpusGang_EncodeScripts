import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import havsfunc as hvf
import awsmfunc as awf
import mvsfunc as mvf
from vsutil import plane
from rekt import *
from adptvgrnMod import *

def deband(clip, range=15, y=64, cb=32, cr=32, ygr=1, cgr=.5, size=1, sharp=50, static=False, luma_scaling=5,
           mask="retinex", mmax=2, minf=2, tv_range=True, showmask=False) -> vs.VideoNode:
    dbn = clip.f3kdb.Deband(range=range, y=y, cb=cb, cr=cr, grainy=0, grainc=0, output_depth=16)

    if mask is not None:
        from vsutil import iterate
        if mask == "retinex":
            import kagefunc as kgf
            msk = kgf.retinex_edgemask(clip)
        elif mask == "debandmask":
            from debandmask import debandmask
            msk = debandmask(clip.std.ShufflePlanes(0, format=vs.GRAY))
        else:
            msk = mask(clip.std.ShufflePlanes(0, format=vs.GRAY))
        msk = iterate(msk, core.std.Maximum, mmax)
        msk = iterate(msk, core.std.Inflate, minf)
        if showmask:
            return msk
        dbn = core.std.MaskedMerge(dbn, clip, msk)

    grn = adptvgrnMod(dbn, strength=ygr, cstrength=cgr, size=size, sharp=sharp, static=static,
                      luma_scaling=luma_scaling)

    if tv_range:
        return grn.std.Expr("x 4096 < x 4096 - abs 4096 + x ?").resize.Point(format=clip.format)
    else:
        return grn.resize.Point(format=clip.format)


def spltscrn(clip, left=0, top=0, width=1280, height=720, fix="bbmod") -> vs.VideoNode:
    import awsmfunc as awf
    if left > 0:
        lft = clip.std.Crop(right=clip.width - left)
        lft = awf.bbmod(lft, right=1) if fix == "bbmod" else lft.cf.ContinuityFixer(right=1)
        lft = awf.cr(lft, height=height)
        rht = clip.std.Crop(left=left)
        rht = awf.bbmod(rht, left=1) if fix == "bbmod" else rht.cf.ContinuityFixer(left=1)
        rht = awf.cr(rht, height=height)
        return core.std.StackHorizontal([lft, rht])
    elif top > 0:
        tpc = clip.std.Crop(bottom=clip.height - top)
        tpc = awf.bbmod(tpc, bottom=1) if fix == "bbmod" else tpc.cf.ContinuityFixer(bottom=1)
        tpc = awf.cr(tpc, width=width)
        btc = clip.std.Crop(top=top)
        btc = awf.bbmod(btc, top=1) if fix == "bbmod" else btc.cf.ContinuityFixer(top=1)
        btc = awf.cr(btc, width=width)
        return core.std.StackVertical([tpc, btc])


src = core.ffms2.Source("video.mkv")

core.max_cache_size = 16000

src = fvf.Depth(src, 16)

# Dirty line fixes

# Sky issue.  No point in adjusting further than masked FillBorders.  Matching grain is a waste of time for this shit.
mask = core.std.ShufflePlanes(src, 0, vs.GRAY).std.Binarize(43500)
cf = core.fb.FillBorders(src, top=6, mode="mirror").std.MaskedMerge(src, mask)
bb = fvf.ReplaceFramesSimple(src, cf, mappings="[17495 17533]")

# Dots in top and bottom left corners.
mask = core.std.ShufflePlanes(src, 0, vs.GRAY).std.Binarize(25000).std.Invert().std.Maximum().std.Inflate().std.Invert()
cf = core.fb.FillBorders(src, left=2).std.MaskedMerge(bb, mask)
bb = fvf.ReplaceFramesSimple(bb, cf, mappings="[42479 42497]")

# Four pixel wide dirty line across half the left border that gets darker towards the bottom.
cf = core.cf.ContinuityFixer(src, left=4, right=0, bottom=0, top=0, radius=20).std.Crop(top=1080 - 400)
top = core.std.Crop(src, bottom=400)
cf = core.std.StackVertical([top, cf])
bb = fvf.ReplaceFramesSimple(bb, cf, mappings="[90727 90816]")

# Different flashback/grain scenes.
cf = awf.bbmod(src, cTop=1, cBottom=0, cLeft=2, cRight=2)
cf = cf.cf.ContinuityFixer(left=[2, 2, 2], right=0, bottom=0, top=0, radius=1000)
fb = cf.fb.FillBorders(left=2, mode="fillmargins")
mask = core.std.ShufflePlanes(src, 2, vs.GRAY).std.Binarize(27000).std.Invert().resize.Bilinear(1920, 1080)
cf = core.std.MaskedMerge(cf, fb, mask, planes=[1, 2])
bb = fvf.ReplaceFramesSimple(bb, cf, mappings="[5571 5730] [104581 105166]")

cf = core.cf.ContinuityFixer(src, left=[2, 0, 2], top=1, right=[2, 0, 2], bottom=0, radius=5)
cf = core.fb.FillBorders(src, left=2, right=2, top=0, bottom=0, mode="fillmargins").std.Merge(clipb=cf, weight=[1, 0, 0])
bb = fvf.ReplaceFramesSimple(bb, cf, mappings="[105323 105406] [106427 106474]")

cf = core.cf.ContinuityFixer(src, left=[2, 0, 0], top=[2, 0, 0], right=[2, 0, 0], bottom=[2, 0, 0], radius=5)
cf = core.fb.FillBorders(src, left=2, right=0, top=0, bottom=0, mode="fillmargins").std.Merge(clipb=cf, weight=[1, 0, 0])
bb = fvf.ReplaceFramesSimple(bb, cf,
                             mappings="[106767 108038] [105529 105610] [106475 106618] [106671 108038] [108503 109904]")

cf = core.cf.ContinuityFixer(src, left=[2, 2, 2], top=[2, 2, 2], right=[2, 2, 2], bottom=0, radius=5)
cf = core.fb.FillBorders(src, left=2, right=0, top=0, bottom=0, mode="fillmargins").std.Merge(clipb=cf, weight=[1, 0, 0])
bb = fvf.ReplaceFramesSimple(bb, cf, mappings="[105407 105472] [104615 104698] [105611 106426] [106619 106670]")

# TV and screen fixes.  First two are commented out since they seem to be intentional.
# cf=core.fb.FillBorders(bb,left=7,right=7,top=2,mode="fillmargins")
# cf=core.std.Merge(bb,cf,[0,1]).cf.ContinuityFixer(left=[10,0,0],right=[5,0,0],top=0,bottom=0,radius=10)
# bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[137359 137508]")
#
# cf=core.fb.FillBorders(bb,left=5,right=5,top=4,bottom=6,mode="fillmargins")
# cf=core.std.Merge(bb,cf,[0,1]).cf.ContinuityFixer(left=[5,0,0],right=[5,0,0],top=0,bottom=0,radius=10)
# bb=fvf.ReplaceFramesSimple(bb,cf,mappings="[9122 9229]")

cf = core.cf.ContinuityFixer(src, left=[0, 0, 0], top=[1, 0, 0], right=[1, 0, 0], bottom=[1, 0, 0], radius=5)
cf = core.fb.FillBorders(src, left=2, right=0, top=2, bottom=2, mode="fillmargins").std.Merge(clipb=cf, weight=[1, 0, 0])
bb = fvf.ReplaceFramesSimple(bb, cf, mappings="[44624 44696]")

# More complicated TV fix.  Bottom is partially damaged and hard to mask around.
fb = core.fb.FillBorders(src, bottom=2)
luma = core.std.ShufflePlanes(src, 0, vs.GRAY).std.Crop(top=1079)
top = core.std.BlankClip(src, height=1079, width=1920, format=vs.GRAY16)
layer_a = luma.std.Invert().std.Binarize(40000)
layer_b = core.std.Expr([layer_a, luma.std.Invert().std.Binarize(30000)], "y x -")
layer_c = luma.std.Binarize(40000)
mask_a = core.std.StackVertical([top, layer_a])
mask_b = core.std.StackVertical([top, layer_b])
mask_c = core.std.StackVertical([top, layer_c])
cf_a = awf.FixRowBrightnessProtect2(bb, 1079, 50)
cf_b = fb  # awf.FixRowBrightnessProtect2(bb,1079,50)
cf_c = core.cf.ContinuityFixer(src, left=0, top=0, right=0, bottom=3, radius=15)
merge_a = core.std.MaskedMerge(src, cf_a, mask_a)
merge_b = core.std.MaskedMerge(merge_a, cf_b, mask_b)
merge_c = core.std.MaskedMerge(merge_b, cf_c, mask_c)
cf = merge_c.cf.ContinuityFixer(left=0, top=0, right=0, bottom=2, radius=15)
cf = core.std.Merge(cf, fb, [0, 1])
mask = core.std.StackVertical([top, luma.std.Binarize(24500)])
cf = core.std.MaskedMerge(cf, fb, mask)
mask = core.resize.Spline36(core.std.ShufflePlanes(cf, 1, vs.GRAY), 1920, 1080).std.Binarize(34000)
cf = core.std.MaskedMerge(cf, fb, mask)
sf = fvf.ReplaceFramesSimple(bb, cf, mappings="[127071 127118]")

# UHD and tint

clip = core.resize.Point(sf, matrix_in_s="709", format=vs.RGBS)
cubed = core.timecube.Cube(clip, "greentopink.cube")
sf = core.resize.Spline36(cubed, 1280, 720, matrix=1, format=vs.YUV420P16, dither_type="error_diffusion")
rs = sf

uhd = core.ffms2.Source("yournameuhdsegment1.mkv").cf.ContinuityFixer(2, 2, 2, 2, 20)
uhd = core.std.AssumeFPS(uhd, fpsnum=24).std.Trim(46, 213)
uhd = fvf.Depth(uhd, 16)
uhd = awf.DynamicTonemap(uhd).resize.Point(matrix_in_s="709", format=vs.RGBS)
uhd = core.timecube.Cube(uhd, "yournamehud.cube").resize.Point(matrix_s="709", format=vs.YUV420P16, dither_type="error_diffusion")
uhd = uhd.resize.Spline36(1280, 720)
sf = sf.std.Trim(0, 48653) + uhd + sf.std.Trim(48822, 153317)
#sf = sf.std.Trim(0, 48653) + core.std.Interleave([awf.FrameInfo(uhd, "UHD Tonemapped"), awf.FrameInfo(sf, "HD Reference").std.Trim(48654, 48821)]) + sf.std.Trim(48822, 153317)


spl = spltscrn(rs, left=958)
sf = fvf.rfs(sf, spl, "[46887 46934]")

spl = spltscrn(rs, top=538)
sf = fvf.rfs(sf, spl, "[42966 43001] [45052 45117]")

spl = spltscrn(rs, left=960)
sf = fvf.rfs(sf, spl, "[42938 42965]")

rs = src
ypl = plane(rs, 0)
upl = plane(rs, 1)
vpl = plane(rs, 2)

yss = spltscrn(ypl, 959, fix=0)
uss = spltscrn(upl, 480, height=360, fix=0)
vss = spltscrn(vpl, 480, height=360, fix=0)
spl = core.std.ShufflePlanes([yss, uss, vss], [0, 0, 0], vs.YUV)
sf = fvf.rfs(sf, spl, "[46935 46998]")


# Single columns missing left and right.  Chroma missing on right border.
cf = core.fb.FillBorders(src, left=1, right=1, mode="fillmargins")
cf = core.fb.FillBorders(cf, left=2, mode="fillmargins").std.Merge(cf, [1, 0])
#cf = awf.cr(cf, 1280, left=1, right=1)
cf = cf.resize.Spline36(1280, 720, src_left=1, src_width=1918)
sf = fvf.ReplaceFramesSimple(sf, cf,
                             mappings="[46584 46619] [64834 64869] [37247 37366] [54042 54257] [54318 54437] [71127 71174] [75005 75052] [79985 80128] [112235 112270] [116463 116498] [116555 116644] [117732 117761] [135403 135480] [143575 143586] [143599 143610]")

# One missing and one heavily damaged border on the right.
cf = core.fb.FillBorders(src, right=1, mode="fillmargins").cf.ContinuityFixer(left=0, right=2, bottom=0, top=0,
                                                                                    radius=150)
cf = awf.cr(cf, 1280, 720, right=1)
sf = fvf.ReplaceFramesSimple(sf, cf, mappings="43903")

jpn = core.ffms2.Source("Your.Name.2016.BluRay.Remux.1080p.AVC.DTS HD.MA.5.1-HiFi.mkv")
jpn = core.std.AssumeFPS(jpn, fpsnum=24)
jpn = awf.DelFrameProp(jpn)
jpn = fvf.Depth(jpn, 16)
jpn = jpn.resize.Spline36(1280, 720)
sf = fvf.rfs(sf, jpn, "[2412 2813] [62555 62841]")

# Debanding

dbn = deband(sf, range=15, y=48, cb=52, cr=52, ygr=0.5, cgr=0.2, static=True)
sf = fvf.ReplaceFramesSimple(sf, dbn, mappings="[480 516] [38915 38930] [132848 132871]")

dbn = deband(sf, range=30, y=0, cb=0, cr=54, ygr=0.1, cgr=0.2, static=True)
sf = fvf.rfs(sf, dbn, "[128483 128572]")

dbn = deband(sf, range=15, y=54, cb=16, cr=16, ygr=0.2, cgr=.05, static=True)
sf = fvf.ReplaceFramesSimple(sf, dbn,
                             mappings="[291 450] [1056 1247] [3630 3634] [9350 9397] [9578 9592] [46044 46178] [46935 46998] [65084 65131] [75701 75988] [99862 99875] [103862 103866][130892 130951] [131181 131312]")

grn = adptvgrnMod(sf, strength=.2, cstrength=.05, size=1, static=True, luma_scaling=9)
mask = core.std.Sobel(src.std.ShufflePlanes(0, vs.GRAY).resize.Bilinear(1280, 720)).std.Binarize(15000).std.Maximum().std.Inflate()
sf = core.std.MaskedMerge(grn, sf, mask)

# Blacks

mask = core.std.ShufflePlanes(sf, 0, vs.GRAY).std.Binarize(15000).std.Inflate().std.Maximum().std.Inflate()
black = core.std.BlankClip(sf, 1280, 720, format=vs.YUV420P16, color=[4096, 32768, 32768])
merge = core.std.MaskedMerge(black, sf, mask)
sf = fvf.rfs(sf, merge,
             " [0 26] [265 289] [451 480] [1248 1367] [2184 2255] [2679 2867] [4810 4944] [24924 24994] [38931 39030] [46999 47117] [63232 63387] [91525 91643] [132872 132991] [133448 133590] [141890 141959] [145638 153317]")

mask = core.std.ShufflePlanes(sf, 0, vs.GRAY).std.Binarize(4500).std.Inflate()
merge = core.std.MaskedMerge(black, sf, mask)
sf = fvf.rfs(sf, merge, "[290 450] [53790 53873]")

out = mvf.Depth(sf, 8, dither=7)

out.set_output()