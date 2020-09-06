import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
from rekt import *
core = vs.get_core()

#1 Import
src = core.ffms2.Source('/home/Videos/taipei/remux.mkv')

#2 Crop
src = src.std.Crop(top=20, bottom=20)
src = core.fb.FillBorders(src, left=0, right=0, top=1, bottom=1, mode="fillmargins")

#3 Dirty Lines
bb = awf.bbmod(src, left=0, right=0, top=2, bottom=0, thresh=None, blur=50, y=True, u=True, v=True)
bb = awf.bbmod(bb, left=0, right=0, top=0, bottom=2, thresh=None, blur=5, y=True, u=True, v=True)
bb = awf.bbmod(bb, left=1, right=1, top=0, bottom=0, thresh=None, blur=75, y=True, u=True, v=True)

#4 Resize and crop the filled line
resize = awf.CropResize (clip=bb, width = 1280, left=0, right=0, top=1, bottom=1)

#5 Select extract
#extract = awf.SelectRangeEvery(clip=resize, every=4000, length=50, offset=10000)

resize.set_output()
