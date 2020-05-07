import vapoursynth as vs
core = vs.get_core()

src = core.ffms2.Source("")
src = src.std.Crop(top=138, bottom=138)
src.set_output()
