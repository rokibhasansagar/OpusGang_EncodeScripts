import vapoursynth as vs
import awsmfunc as awf
import fvsfunc as fvf
#import kagefunc as kgf
from rekt import *
#rom adptvgrnMod import *
core = vs.get_core()

#1 Import
src = core.ffms2.Source('/home/Videos/asako/remux.mkv')

#2 Crop
src = src.std.Crop(left=60, right=60, top=0, bottom=0)

#3 Fix edge levels and balance them
lvl = rektlvls(src, rownum=1079, rowval=-1, colnum=[0,1,1799], colval = [-1,-1,-1], prot_val=0)
bb = awf.bbmod(src, top=1, thresh=None, blur=130, y=True, u=True, v=True)
bb = awf.bbmod(src, bottom=1, thresh=None, blur=100, y=True, u=True, v=True)
bb = awf.bbmod(src, left=2, thresh=None, blur=15, y=True, u=True, v=True)
bb = awf.bbmod(src, right=2, thresh=None, blur=20, y=True, u=True, v=True)

#4 Select extract
#extract = awf.SelectRangeEvery(clip=bb, every=1500, length=48, offset=15000)

bb.set_output()
