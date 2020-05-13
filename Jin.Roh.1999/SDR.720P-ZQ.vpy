import vapoursynth as vs
import awsmfunc as awsm
import fvsfunc as fvf
import adptvgrnMod as agm
import havsfunc as hvf
from rekt import rektlvls, rekt_fast

core = vs.core
src = core.lsmas.LWLibavSource(r'00000_jpn.m2ts')
src = core.std.Crop(src, left = 44, right = 46, top = 44, bottom = 44)
b16 = fvf.Depth(src, 16)

dln = rektlvls(b16, colnum = 0, colval = 64, rownum = [0,1,990,991], rowval = [50,50,50,50])
dln = rekt_fast(dln, lambda x: core.fb.FillBorders(x, left = 1), bottom = 990)
dln = rekt_fast(dln, lambda x: core.fb.FillBorders(x, left = 1), top = 990)
dln = awsm.bbmod(dln, right = 1)
dln = awsm.bbmod(dln, y = False, left = 2, top = 2, bottom = 2)

resize = core.resize.Spline36(dln, 1280, 694, dither_type = "error_diffusion")

luma = core.std.ShufflePlanes(resize, 0, vs.GRAY)
denoise = core.knlm.KNLMeansCL(luma, h = 2)
diff = core.std.MakeDiff(luma, denoise)
RG = hvf.STPresso(diff, limit = 6, bias = 72, tbias = 55, tthr = 24)
merge = core.std.MergeDiff(denoise, RG)
merge = core.std.ShufflePlanes([merge, resize], [0,1,2], vs.YUV)

final = fvf.Depth(merge, 8)
final.set_output()