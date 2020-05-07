import vapoursynth as vs
core = vs.get_core()
import detect as dtc
import fvsfunc as fvf
import awsmfunc as awf
import havsfunc as haf
from adptvgrnMod import *
from vsutil import *

src = core.ffms2.Source("/home/aicha/Encoding/dark_waters/Dark.Waters.2019.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-decibeL.mkv")

src = fvf.Depth(src, 16)

src = src.std.Crop(top=138, bottom=138)

# top chroma has a dirty line, split to speed things up
y, u, v = split(src)
u = awf.fb(u, top=1)
v = awf.fb(v, top=1)

src = join([y, u, v])

# dirty lines on first intro
bbm = awf.bbmod(src, left=4, top=2, right=7, blur=999, y=True, u=False, v=False)
snf = awf.rfs(src, bbm, "[35 320]")

# banding during intros and one fade
dbn = src.f3kdb.Deband(range=20, y=64, cb=0, cr=0, grainy=0, grainc=0, output_depth=16)
grn = adptvgrnMod(dbn, size=1, static=True, fade_edges=True, strength=.2)
snf = awf.rfs(snf, grn, "[350 452]")

grn = adptvgrnMod(dbn, static=False, fade_edges=True, strength=.3, size=1.8, sharp=55)
snf = awf.rfs(snf, grn, "[35 320] [92496 92503]")

# ringing on texts during opening sequence
drn = haf.HQDeringmod(src, show=False, mrad=2, drrep=0, mthr=255)
lms = get_y(src).std.Binarize(100 << 8)
lms = iterate(lms, core.std.Maximum, 4)
drn = core.std.MaskedMerge(src, drn, lms)
snf = awf.rfs(snf, drn, mappings="[2156 2252] [2481 2564] [2622 2705] [2757 2840] [2919 3002] [3043 3126] [3644 3727] [3825 3909] [4516 4599] [4691 4774] [4921 5004] [5080 5164] [5291 5450] [5515 5677] [5863 5952] [6021 6104]")

out = fvf.Depth(snf, 8)

out.set_output()