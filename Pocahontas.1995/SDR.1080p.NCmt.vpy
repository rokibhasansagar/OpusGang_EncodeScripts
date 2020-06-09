import vapoursynth as vs
import awsmfunc as awf
import sgvsfunc as sgf
import kagefunc as kgf
import fvsfunc as fvf
from rekt import rektlvls
from vsutil import depth
core = vs.get_core()

#source
src= core.ffms2.Source('/home/f00d/scratch/encoding/Pocahontas.1995.TWN.BluRay.Remux.1080p.AVC.DTS-HD.MA.5.1-ReQuEsT.mkv')

b16=depth(src,16)

msk = kgf.retinex_edgemask(core.knlm.KNLMeansCL(b16, h=1)).std.Maximum()
db = core.f3kdb.Deband(b16, y=60, cb=60, cr=60, grainy=32, grainc=32, dynamic_grain=True, range=16)
mrg = core.std.MaskedMerge(db, b16, msk)

final = mrg
final = depth(final, 8)

final.set_output()