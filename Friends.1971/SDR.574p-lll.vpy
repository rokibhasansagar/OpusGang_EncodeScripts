import vapoursynth as vs
import rekt
core = vs.get_core()


src = core.d2v.Source('/home/f00d/scratch/encoding/movie.d2v')
crop = core.std.Crop(src, left=0, right=0, top=0, bottom=2)
filter = rekt.rekt_fast(crop, fun=lambda m: awf.saturation(m, sat=1.65), right=406, bottom=572)
filter = rekt.rekt_fast(crop, fun=lambda m: core.fb.FillBorders(m, top=1, mode="fillmargins"), right=404, bottom=570)

filter.set_output()