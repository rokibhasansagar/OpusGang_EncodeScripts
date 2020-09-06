import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from rekt import *
from adptvgrnMod import *
core = vs.get_core()

#1 Import
src = core.ffms2.Source('/home/Videos/beach/remux.mkv')

#2 Crop and change bit depth
src = src.std.Crop(top=20, bottom=20)
src = fvf.Depth(src, 16)

#3 Dirty Lines
src = rektlvls(src, rownum=0, rowval=7)
src = awf.bbmod(src, bottom=2, u=False, v=False)

# Debanding
dbn = core.f3kdb.Deband(src, range=18, y=64, cb=45, cr=40, grainy=0, grainc=0, output_depth=16)
grn = adptvgrnMod(dbn, size=1.1, strength=1.0, cstrength=0.4, luma_scaling=3)
final = fvf.rfs(dbn,grn, "[1394,1465]")
out = fvf.Depth(final, 8)

out.set_output()
