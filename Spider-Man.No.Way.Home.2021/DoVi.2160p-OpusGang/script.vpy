import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

core.num_threads = 8

src_path = Path(
    "Spider-Man.No.Way.Home.2021.UHD.BluRay.2160p.TrueHD.Atmos.7.1.DV.HEVC.REMUX-FraMeSToR.mkv"
)
src = core.ffms2.Source(src_path)
src = src.std.Crop(top=276, bottom=276)

src.set_output()
