import vapoursynth as vs
core = vs.get_core()
from bandmask import *
import awsmfunc as awf
from adptvgrnMod import *
import fvsfunc as fvf
import havsfunc as haf
from debandmask import *
from vsutil import *
import vsTAAmbk as taa

src = core.ffms2.Source("/home/aicha/Encoding/gintama/Gintama 2017 2160p UHD Blu-ray Remux HEVC HDR10 DTS-HD MA 5.1-GINTAMA.mkv")

src = fvf.Depth(src, 16)

dbl = fvf.Debilinear(src, 1920, 1080)

blk = src.std.BlankClip(1920, 1080, color=[4096, 32768, 32768], format=vs.YUV420P16)
snf = awf.rfs(dbl, blk, "[0 26] [10152 10155]")

op1 = src.std.Crop(top=42, bottom=42)
op1 = awf.bbmod(op1, top=1, bottom=2)
op1 = fvf.Debilinear(op1, 1920, 1038)
op1 = op1.neo_f3kdb.Deband(range=20, y=80, cb=80, cr=80, grainy=0, grainc=0, sample_mode=4, output_depth=16)
op1 = adptvgrnMod(op1, strength=1/3, static=True, luma_scaling=5)
op1 = op1.std.AddBorders(top=20, bottom=22)
op2 = src.std.Crop(top=44, bottom=44, left=2, right=2)

op2 = awf.bbmod(op2, left=4, right=4).resize.Spline36(1920, 1036).std.AddBorders(top=22, bottom=22)
snf = awf.rfs(snf, op1, "[27 348] [10156 10475]")
snf = awf.rfs(snf, op2, "[349 647]")

aan = haf.HQDeringmod(dbl)
aan = taa.TAAmbk(aan, aatype=2, opencl=True)
aan = haf.FineDehalo(aan)
snf = awf.rfs(snf, aan, "[7514 9558] [9682 10145]")

dbn = dbl.neo_f3kdb.Deband(range=15, y=64, cb=0, cr=0, grainy=0, grainc=0, sample_mode=4, output_depth=16)
msk = debandmask(dbl).std.Inflate().std.Inflate()
mrg = core.std.MaskedMerge(dbn, dbl, msk)
snf = awf.rfs(snf, mrg, "[648 736] [11358 11427] [11715 11794] [13175 13261] [29844 29963] [29964 30067] [180928 181103]")

y, u, v = split(snf)
umk = bandmask(u, thr=50)
vmk = bandmask(v, thr=50)
udb = u.neo_f3kdb.Deband(range=10, y=64, grainy=8, sample_mode=4, output_depth=16)
vdb = v.neo_f3kdb.Deband(range=10, y=64, grainy=8, sample_mode=4, output_depth=16)
u = core.std.MaskedMerge(u, udb, umk)
v = core.std.MaskedMerge(v, vdb, vmk)
snf = join([y, u, v])

snf = haf.FineDehalo(snf, rx=1.5, showmask=0, thmi=48, thma=128)

out = fvf.Depth(snf, 10)

out = adptvgrnMod(out, strength=.2, cstrength=.1, luma_scaling=100, show_mask=False, static=False)

out.set_output()