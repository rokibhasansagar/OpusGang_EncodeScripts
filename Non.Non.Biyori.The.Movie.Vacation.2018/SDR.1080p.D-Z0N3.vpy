import vapoursynth as vs
core = vs.get_core()
import fvsfunc as fvf
import kagefunc as kgf
import vsTAAmbk as taa
from rekt import *

src = core.ffms2.Source("")

b16 = fvf.Depth(src, 16)

# Very light denoising.
b16 = kgf.hybriddenoise(b16, knl=0.5, sigma=0.3)

# Placebo edgemask binarized so we only get the obvious edges, then inflated.
mask = kgf.retinex_edgemask(b16).std.Binarize(65500).std.Maximum().std.Inflate()

# Strong aliasing.
aa = taa.TAAmbk(b16, aatype=2, mtype=0, opencl=True)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(b16, aa, mappings="[791 945] [2256 2295] [7881 7958] [11100 11261] [19002 19060] [20028 20164] [20343 20413] [30958 31064] [86001 86102] [87894 87980]")

# Very strong aliasing.
aa = taa.TAAmbk(b16, aatype=6, mtype=0, opencl=True)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[1232 1375] [11495 11622] [20553 20669] [21163 21221] [42004 42074] [42989 43515] [75083 75183]")

# Normal aliasing.
aa = taa.TAAmbk(b16, aatype=3, preaa=0, mtype=0, opencl=True)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[28031 28155] [28968 29042] [43742 43943] [44134 44270] [45545 45857] [76453 76691] [83253 83388]")

# Strong aliasing on diagonal lines.
mask = kgf.retinex_edgemask(b16).std.Maximum(coordinates=[0,1,0,0,0,0,1,0]).std.Binarize(65500).std.Minimum(coordinates=[1,0,1,1,1,1,0,1]).std.Inflate()
aa = taa.TAAmbk(b16, aatype=2, mtype=0, opencl=True)
aa = core.std.MaskedMerge(b16, aa, mask)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[1780 1930] [15159 15538]")

# Very strong aliasing in specific areas of the screen.
aa = rektaa(b16, 1312, 608, 568, 182, aatype=6, mtype=2, opencl=True)
sf = fvf.rfs(sf, aa, "[4433 4569]")

lumamask = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Binarize(8500).std.Minimum().std.Deflate().std.Minimum()
aa = rektaa(b16, 0, 616, 1880, 420, aatype=6).std.MaskedMerge(b16, lumamask)
sf = fvf.rfs(sf, aa, "[14724 14794] [14896 15159]")

aa = rektaa(b16, 700, 484, 1196, 536, aatype=6)
aa = rektaa(aa, 1648, 424, 254, 502, aatype=3, opencl=True)
sf = fvf.rfs(sf, aa, "[32774 32844]")

aa = taa.TAAmbk(b16, aatype=6, mtype=0)
aa = core.std.MaskedMerge(b16, aa, mask)
lumamask = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Binarize(30000).std.Maximum().std.Inflate().std.Maximum().std.Inflate()
aa = core.std.MaskedMerge(b16, aa, lumamask)
aa = rekt(aa, b16, left=500, right=500)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[35416 35498]")

aa = rektaa(b16, 790, 636, 300, 372, aatype=6)
sf = fvf.ReplaceFramesSimple(sf, aa, mappings="[42989 43515]")

final = fvf.Depth(sf, 8)

final.set_output()