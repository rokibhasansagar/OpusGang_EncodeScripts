import vapoursynth as vs
core = vs.core
import awsmfunc as awsm
from vsutil import depth

src = core.ffms2.Source('holymountain.mkv').std.Crop(top=130, bottom=130)
src = depth(src, 16)
fill = awsm.fb(src, top=1, bottom=1)
final = depth(fill, 8)
final.set_output()
