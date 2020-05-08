import vapoursynth as vs
from detect import *
from adptvgrnMod import *
from adptvdbn import *
from rekt import *
from debandmask import *
import fvsfunc as fvf
import havsfunc as haf
import kagefunc as kgf
core = vs.get_core()

src = core.ffms2.Source('Dogville.2003.DEU.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-BMF.mkv') #577
web = core.ffms2.Source('Dogville.2003.1080p.WEBRip.DD2.0.x264-FOCUS.mkv') #431

web_crp = core.std.CropRel(web, left=0, top=138, right=0, bottom=138)
blank1 = core.std.BlankClip(width=1920, height=804, format=vs.YUV420P8, length=146)
blank2 = core.std.BlankClip(width=1920, height=804, format=vs.YUV420P8, length=40)
web_sync = blank1 + web_crp
web_sync = web_sync + blank2
ypl = web_sync.std.ShufflePlanes(0, vs.GRAY).std.Expr("x 70 > x 1.3 * x ?")
color = core.std.BlankClip(width=1920, height=804, format=vs.YUV420P8, length=256212, color=[209, 99, 145])
black = core.std.BlankClip(width=1920, height=804, format=vs.YUV420P8, length=256212)
web_fix = core.std.MaskedMerge(black, color, ypl)


crp = core.std.CropRel(src, left=0, top=138, right=0, bottom=138)
fxb = rektlvls(crp, rownum=[0, 803], rowval=[1.5, 1.5], colnum=[0, 1, 1919], colval=[4.25, 0.5, 0.5])


hybrid = core.remap.ReplaceFramesSimple(fxb, web_fix, mappings="[577 773] [796 966] [13297 13470] [41097 41262] [65813 65978] [82985 83095] [92515 92639] [127513 127670] [140186 140403] [177405 177601] [216486 216667]")
hybrid = fvf.Depth(hybrid, 16)


dbn = adptvgrnMod(core.f3kdb.Deband(hybrid, range=15, y=64, cb=64, cr=64, grainy=0, grainc=0, output_depth=16), strength=.6, size=1.1, show_mask=False)
retinex = kgf.retinex_edgemask(hybrid).std.Binarize(65000).std.Maximum().std.Inflate()
dbn = core.std.MaskedMerge(dbn, hybrid, retinex)

ypl = hybrid.std.ShufflePlanes(0, vs.GRAY)
scl = (ypl.format.bits_per_sample - 8)
msk = ypl.std.Binarize(40 << scl, 255 << scl, 0)

out = core.std.MaskedMerge(hybrid, dbn, msk)
out = fvf.Depth(out, 8)

out.set_output()