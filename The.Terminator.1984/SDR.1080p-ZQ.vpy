import vapoursynth as vs
import awsmfunc as awf
import kagefunc as kgf
import fvsfunc as fvf
from rekt import rektlvls
from vsutil import depth
from awsmfunc import banddtct
core = vs.get_core()



src= core.ffms2.Source('/home/f00d/scratch/encoding/The.Terminator.1984.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-HiFi.mkv')
src = core.std.Crop(src, top=20, bottom=20)

#single line fix, top bottom
#get the level as close as reasonable to source
lvl1 = rektlvls(src, rownum = [0, 1039], rowval = [24, 24], prot_val=0)

#2 pixel BB pass ontop of lvl fix
#without this the chroma is consistently dirty for these two lines
bb = awf.bbmod(lvl1, left = 0, right = 0, top = 2, bottom = 2, thresh = [128, 128, 128], blur = [20, 20, 20], scale_thresh=False,cpass2=False)

#intro and credits are from a different source and don't have the issue the main film does
final = awf.rfs(src, bb, "[846 149089]")

final.set_output()