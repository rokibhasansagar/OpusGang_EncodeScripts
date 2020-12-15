import vapoursynth as vs
import vsutil
import awsmfunc
import fvsfunc
import dysfunctional
from adptvgrnMod import *
import mvsfunc
import havsfunc
import vardefunc
import atomchtools
import lvsfunc
from rekt import rekt
import vsTAAmbk as taa
from kagefunc import kirsch
core = vs.core

height = 844
width = vsutil.get_w(height)

OPS, EDS = 3453, 31553

def shitEdgeCleaner(clip: vs.VideoNode, edgemask, modifier=None, **args) -> vs.VideoNode:
    from vsutil import iterate

    y = vsutil.get_y(clip)
    edgem = edgemask(y, **args)

    if modifier != None:
        edgem = iterate(edgem, lambda x: core.std.Minimum(x), modifier[0])
        edgem = iterate(edgem, lambda x: core.std.Maximum(x), modifier[1])
        edgem = iterate(edgem, lambda x: core.std.Inflate(x), modifier[2])
        
    dft = core.dfttest.DFTTest(y, sigma=5, tbsize=1, planes=[0])

    mrg = core.std.MaskedMerge(y, dft, edgem)
    return core.std.ShufflePlanes([mrg, clip], [0,1,2], vs.YUV)

ntb = core.ffms2.Source(r'Dorohedoro.S01E01.1080p.NF.WEB-DL.DDP2.0.H.264-NTb.mkv')[240:]
br = core.ffms2.Source(r'[Bakaiser] Dorohedoro - 01 [BD 1920x1080 x264 FLAC].mkv')

src = core.ffms2.Source(r'00000.m2ts')
b16 = vsutil.depth(src, 16)
des = fvsfunc.DescaleAA(b16, w=width, h=height, showmask=False, thr=10)

y = vsutil.get_y(des)
msk_1 = dysfunctional.FDOG(des, retinex=True, div=[2,2]).std.Maximum()
msk_2 = atomchtools.Tp7DebandMask(des)
msk_3 = core.std.Sobel(y)
msk_4 = core.std.Binarize(y, 20 << 8)
msk_4 = vsutil.iterate(msk_4, lambda x: core.std.Minimum(x), 12)
msk_5 = kirsch(y)
msk_6 = core.std.Expr([msk_4, msk_5], "x y +").std.Minimum()

rng = havsfunc.HQDeringmod(des, thr=16)
hlo = havsfunc.EdgeCleaner(rng, 20)
mrg = core.std.MaskedMerge(hlo, des, msk_6)
crd = lvsfunc.rfs(des, mrg, ranges=[(OPS+0, OPS+2157)])

aa_1 = taa.TAAmbk(crd, 4, 0, 0)
aa_1 = core.std.MaskedMerge(crd, aa_1, msk_4)
aa_1 = lvsfunc.rfs(crd, aa_1, ranges=[(OPS+414, OPS+2157)])
# maybe later
#dbl = shitEdgeCleaner(des, lambda x: dysfunctional.FDOG(x), modifier=[0,3,0])

dns_y = mvsfunc.BM3D(vsutil.get_y(aa_1), sigma=[1,0,0])
dns_c = core.knlm.KNLMeansCL(aa_1, h=.6, channels='UV')
dmrg = core.std.ShufflePlanes([dns_y, dns_c], [0,1,2], vs.YUV)
dns = core.std.MaskedMerge(dmrg, aa_1, msk_1)

dbn = core.f3kdb.Deband(dns, preset='low/nograin', y=35)
dbn = core.std.MaskedMerge(dbn, dns, msk_2)
dbn = awsmfunc.rfs(dns, dbn, "[15200 15200]")

grn = adptvgrnMod(dbn, seed=444, strength=.15, luma_scaling=6, static=True)

src.set_output(1)
des.set_output(2)
aa_1.set_output(3)
grn.set_output(4)
msk_6.set_output(5)