import vapoursynth as vs
import awsmfunc as awf
import vsutil
from rekt import *
from fxborders import *
 
core = vs.core
 
src = core.ffms2.Source('True.Romance.1993.FRA.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-HiFi.mkv')
 
crp = awf.CropResize(src, top=131, bottom=131)
 
fxb = awf.bbmod(crp, top=2, bottom=2, left=1, y=True, u=False, v=False)
fb = awf.fb(fxb, left=1, right=1, planes=[0, 1, 2])
fxb = fixbrdrs(fb, left=True)
bb = awf.bbmod(fxb, left=2, right=2, y=True, u=False, v=False)
out = bb
 
fxb = rektlvls(crp, rownum=[1, 2, 815, 816], rowval=[-4, -4, -4, -4])
fb = awf.fb(fxb, top=1, bottom=1, planes=[0, 1, 2])
fxb = fb.std.Crop(left=2)
area = core.std.Crop(fxb, bottom=814)
fix = awf.bbmod(area, left=4)
fxb = core.std.StackVertical([fix, fxb.std.Crop(top=4)])
bb = awf.bbmod(fxb, left=2, y=True, u=False, v=False)
brd = awf.AddBordersMod(bb, left=2, lsat=1.2)
 
out = awf.rfs(out, brd, "[27518 29776] [41946 46507] [106303 120329]")
 
out.set_output()
