import vapoursynth as vs
import awsmfunc as awf
core = vs.get_core()

#1 Import
src = core.ffms2.Source('/home/Videos/band/remux.mkv')

#2 Crop
crop= core.std.Crop(clip=src, left=0, right=0, top=20, bottom=22)

#3 Fill borders
fill = core.fb.FillBorders(crop, left=0, right=0, top=1, bottom=1, mode="fillmargins")

#4 Apply bbmod
bb = awf.bbmod(fill, top=2, blur=100)

bb.set_output()
