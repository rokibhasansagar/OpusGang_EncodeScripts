import vapoursynth as vs
core = vs.core
core.max_cache_size = 1024
from vsutil import *
import awsmfunc as awf
from adptvgrnMod import *

src = core.ffms2.Source("The.Dark.Knight.2008.EUR.UHD.BluRay.Remux.2160p.HEVC.HDR.DTS-HD.MA.5.1-BMF.mkv")

src = depth(src, 16)

# it's so smooth help mom I'm scared
# have to grain cropped scenes before borders are re-added, since they'll have an effect on frame brightness
agm = lambda c: adptvgrnMod(c, strength=.1, seed=134756, luma_scaling=100, show_mask=False)

# placebo resizing for cropped scenes
crp = src.std.Crop(top=280, bottom=278)
crp = awf.fb(crp, bottom=1)
crp = awf.bbmod(crp, top=1, bottom=2, planes=0, blur=999, thresh=42)
y = crp.resize.Point(format=vs.YUV444P16).std.Crop(bottom=1)
y = y.placebo.Shader("SSimDownscaler.glsl", 1920, 800, filter="mitchell", linearize=0)
u = plane(crp, 1).resize.Spline36(960, 400, src_height=crp.height / 2 - .5, src_left=-.25)
v = plane(crp, 2).resize.Spline36(960, 400, src_height=crp.height / 2 - .5, src_left=-.25)
crp = core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)
crp = agm(crp)
crp = crp.std.AddBorders(top=140, bottom=140, color=[4096, 32768, 32768])

# placebo resizing for uncropped scenes
y = src.placebo.Shader("SSimDownscaler.glsl", 1920, 1080, filter="mitchell", linearize=0)
u = plane(src, 1).resize.Spline36(960, 540, src_left=-.25)
v = plane(src, 2).resize.Spline36(960, 540, src_left=-.25)
src = core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV)
src = agm(src)

# brdrdtct results
src = awf.rfs(src, crp, "[0 1273] [9333 25753] [25896 41837] [42000 45612] [46394 48895] [50057 50742] [51100 51413] [51732 53182] [54182 62673] [62851 79573] [80018 85505] [85665 107049] [119204 131393] [131557 135342] [135440 139123] [140904 151362] [151588 152021] [152059 152530] [152582 153271] [153508 153719] [153763 157362] [157713 159916] [159965 160020] [160286 161070] [161166 161327] [162425 162649] [162800 168406] [168602 180094] [180368 180438] [180520 181608] [183458 183979] [186813 188118] [188723 189518] [189775 190543] [190890 191027] [191791 204921] [207774 218954]")

src = depth(src, 10)

src.set_output()
