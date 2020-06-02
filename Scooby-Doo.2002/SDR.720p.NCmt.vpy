import vapoursynth as vs
import awsmfunc as awf
import sgvsfunc as sgf
from rekt import rektlvls
core = vs.get_core()

#source
src= core.ffms2.Source('/home/f00d/encoding/Scooby-Doo.2002.EUR.BluRay.Remux.1080p.VC-1.DD.5.1-decibeL.mkv')

#crop
crop = core.std.Crop(src, 6, 6, 0, 0)

#fix edge levels and balence them
lvls = rektlvls(crop, rownum = [0, 1, 2, 3, 1076, 1077, 1078, 1079], rowval = [-7, -7, 2, 2, 2, 2, -8, -8], colnum = [0, 1, 1906, 1905], colval = [-5, -3, -2, -1], prot_val=0)
bb = awf.bbmod(lvls, left = 1, right = 2, top = 0, bottom = 0,thresh = [128, 128, 128], blur = [20, 20 ,20], scale_thresh = False, cpass2 = False)
bb2 = awf.bbmod(bb, left = 0, right = 0, top = 2, bottom=2,thresh=[128, 128, 128], blur=[40,40,40], scale_thresh=False,cpass2=False)

#fill the right line which couldn't be fixed
fill = core.fb.FillBorders(bb2, 0, 1, 0, 0, mode="fillmargins")

#resize & crop the filled line
resize = sgf.CropResize(clip = fill, width = 1272, right = 1)

#test = sgf.SelectRangeEvery(clip=resize, every=3000, length=50, offset=10000)

final=resize

final.set_output()