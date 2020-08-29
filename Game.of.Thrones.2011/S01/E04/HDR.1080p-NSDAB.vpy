import vapoursynth as vs
core = vs.core
core.max_cache_size = 2048
import fvsfunc as fvf
import kagefunc as kgf
import awsmfunc as sgf
awf = sgf

src = core.ffms2.Source("Game.of.Thrones.S01E04.Cripples.Bastards.and.Broken.Things.2160p.UHD.BluRay.REMUX.HDR.HEVC.Atmos-EPSiLON.mkv")

src = fvf.Depth(src, 16)

src = fvf.Debilinear(src, 1920, 1080, yuv444=True)

b16 = fvf.Depth(src, 16)

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = kgf.retinex_edgemask(b16)
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = awf.rfs(b16, merged, mappings="[135 191]")

db = b16.f3kdb.Deband(range=15, y=25, cb=0, cr=0, grainy=32, grainc=15, output_depth=16, dynamic_grain=True)
mask = kgf.retinex_edgemask(b16).std.Binarize(10000)
mask2 = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Expr("x 4500 > x 5000 < and x 100 * x 100 / ?")
merged = core.std.MaskedMerge(b16, db, mask2)
merged = core.std.MaskedMerge(merged, b16, mask)
sfdb = awf.rfs(sfdb, merged, mappings="[4434 4607] [8397 8460] [8594 8646] [8696 8778] [8968 9060] [9419 9486] [30009 30222] [30275 30584] [37466 37534] [37585 37644] [76006 76287] [76396 76502] [76622 76886] [77062 77340] [77546 77752] [78002 78077] [78228 78339]")

final = fvf.Depth(sfdb, 10)

final.set_output()
