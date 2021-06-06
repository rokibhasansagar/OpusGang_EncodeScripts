import vapoursynth as vs
from vapoursynth import core
from pathlib import Path

import vsutil
import awsmfunc as awf
import rekt
import havsfunc as haf
import dysfunctional as dysf
import vsTAAmbk
import kagefunc as kgf
import mvsfunc as mvf
import lvsfunc as lvf

from adptvgrnMod import adptvgrnMod

with open(Path("~/Encoding/ssimdown.glsl").expanduser()) as s:
    shader = s.read()

core.num_threads = 8
filter = True

src_path = Path("Tenet.2020.UHD.BluRay.2160p.DTS-HD.MA.5.1.HEVC.REMUX-FraMeSToR.mkv")
src = core.ffms2.Source(src_path)

src = vsutil.depth(src, 16)
#src = src.resize.Spline36(width=1280, height=720)
#awf.detect.brdrdtct(src, top=68, bottom=68, color=[0, 127, 127], color_second=[17, 129, 129])

if filter:
    imax = dysf.ssimdown(src, preset=1080, shader_str=shader)
    imax = awf.bbmod(imax, top=2, bottom=2)

    # Actually top=206, bottom=206
    reg = src.std.Crop(top = 204, bottom = 204)
    reg = awf.fb(reg, top=3, bottom=3)
    reg = awf.bbmod(reg, top=4, bottom=4)
    reg = dysf.ssimdown(reg, preset=1080, shader_str=shader)

    # Actually top=103, bottom=103 but mod2
    reg = reg.std.AddBorders(top=102, bottom=102)

    # Since we undercropped, make the next border row black
    reg = rekt.rektlvls(reg, rownum=[102, 977], rowval=[-100, -100])

    src = awf.rfs(imax, reg, mappings="[0 526] [12602 15729] [18653 25244] [26088 28063] [29005 29087] [29959 30051] [30200 30255] [30465 30950] [31084 34030] [34114 34277] [34339 34353] [34785 38743] [39361 43100] [43131 43265] [43352 43407] [43479 43618] [43699 43808] [43848 46083] [46320 46471] [46884 46990] [47211 47361] [47959 48304] [48436 49385] [49774 49892] [49975 50099] [50297 50353] [50406 50665] [51687 53388] [53439 53498] [53541 55848] [69371 73361] [73814 75308] [76040 77276] [77477 81161] [84971 88735] [88859 89034] [89105 89210] [89347 89394] [89624 89666] [89697 89732] [89785 90646] [90757 93334] [93583 93690] [96487 98421] [98575 99048] [99349 100255] [101130 102822] [103535 105230] [106915 107054] [118619 130782] [136909 138889] [139033 143677] [143834 145112] [152863 159141] [159263 159594] [160214 163354] [163854 164542] [164895 165434] [165477 165507] [165580 167336] [168117 169121] [169527 172166] [175476 176548] [177681 178955] [181398 181758] [183514 184857] [187115 187421] [187831 187881] [188035 188093] [188590 189004] [189226 189290] [189378 189404] [189764 189824] [189947 190074] [190170 190305] [191384 191443] [191543 191608] [191773 191816] [191967 192025] [192123 192165] [192281 192338] [192648 192719] [193355 193664] [194058 194582] [194945 195561] [195896 196552] [200129 205645] [206019 206473] [206561 207266] [207360 216023]")
else:
    src = dysf.ssimdown(src, preset=1080, shader_str=shader)

src = vsutil.depth(src, 10, dither_type='error_diffusion')

src.set_output()