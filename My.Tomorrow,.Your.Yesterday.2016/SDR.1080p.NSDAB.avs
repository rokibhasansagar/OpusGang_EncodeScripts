JPN=FFVideoSource("")
JPNc=Crop(JPN,0,138,0,-138)

THA=FFVideoSource("")
THAc=Crop(THA,0,138,0,-138)

yuy2=ConvertToYUY2(JPNc,ChromaInPlacement="MPEG2", chromaresample="bicubic", matrix="PC.709", interlaced=false)

delogo1=s_ExLogo(yuy2,850,336,42,24,BlurMode=1,LogoMode=2,Amount=255)
art1=ConvertToYV12(delogo1,matrix="PC.709", interlaced=false)
artgrain1=art1.GrainFactory3(g1str=2, g2str=8, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160).Tweak(bright=2.2)
artgrainsh1=StackHorizontal(JPNc.Crop(0,0,-1070,0),artgrain1.Crop(850,0,-1036,0),JPNc.Crop(884,0,0,0))
final1=StackVertical(JPNc.Crop(0,0,0,-468),artgrainsh1.Crop(0,336,0,-444),JPNc.Crop(0,360,0,0))

delogo2=s_ExLogo(yuy2,850,350,34,36,BlurMode=1,LogoMode=2,Amount=255)
art2=ConvertToYV12(delogo2,matrix="PC.709", interlaced=false)
artgrain2=art2.medianblur(radiusy=1, radiusu=1, radiusv=1, calcborder=false).GrainFactory3(g1str=2, g2str=3, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160)
artgrainsh2=StackHorizontal(JPNc.Crop(0,0,-1070,0),artgrain2.Crop(850,0,-1036,0),JPNc.Crop(884,0,0,0))
final2=StackVertical(JPNc.Crop(0,0,0,-454),artgrainsh2.Crop(0,350,0,-414),JPNc.Crop(0,390,0,0))

delogo3=s_ExLogo(yuy2,852,350,34,27,BlurMode=0,LogoMode=2,Amount=255)
art3=ConvertToYV12(delogo3,matrix="PC.709", interlaced=false)
artgrain3=art3.medianblur(radiusy=1, radiusu=1, radiusv=1, calcborder=false).GrainFactory3(g1str=2, g2str=3, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160)
artgrainsh3=StackHorizontal(JPNc.Crop(0,0,-1070,0),artgrain3.Crop(850,0,-1036,0),JPNc.Crop(884,0,0,0))
final3=StackVertical(JPNc.Crop(0,0,0,-454),artgrainsh3.Crop(0,350,0,-414),JPNc.Crop(0,390,0,0))

delogo4=s_ExLogo(yuy2,848,342,34,20,BlurMode=0,LogoMode=2,Amount=255)
art4=ConvertToYV12(delogo4,matrix="PC.709", interlaced=false)
artgrain4=art4.GrainFactory3(g1str=1, g2str=2, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=1.0, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160).Tweak(bright=2.1)
artgrainsh4=StackHorizontal(JPNc.Crop(0,0,-1068,0),artgrain4.Crop(852,0,-1038,0),JPNc.Crop(882,0,0,0))
final4=StackVertical(JPNc.Crop(0,0,0,-462),artgrainsh4.Crop(0,342,0,-442),JPNc.Crop(0,362,0,0))

delogo5=s_ExLogo(yuy2,846,332,36,26,BlurMode=1,LogoMode=2,Amount=255)
art5=ConvertToYV12(delogo5,matrix="PC.709", interlaced=false)
artgrain5=art5.GrainFactory3(g1str=2, g2str=8, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160).Tweak(bright=2.2)
artgrainsh5=StackHorizontal(JPNc.Crop(0,0,-1072,0),artgrain5.Crop(848,0,-1038,0),JPNc.Crop(882,0,0,0))
final5=StackVertical(JPNc.Crop(0,0,0,-472),artgrainsh5.Crop(0,332,0,-452),JPNc.Crop(0,352,0,0))

delogo6=s_ExLogo(yuy2,844,332,38,26,BlurMode=1,LogoMode=2,Amount=255)
art6=ConvertToYV12(delogo6,matrix="PC.709", interlaced=false)
artgrain6=art6.GrainFactory3(g1str=2, g2str=8, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160).Tweak(bright=2.2)
artgrainsh6=StackHorizontal(JPNc.Crop(0,0,-1074,0),artgrain6.Crop(846,0,-1038,0),JPNc.Crop(882,0,0,0))
final6=StackVertical(JPNc.Crop(0,0,0,-472),artgrainsh6.Crop(0,332,0,-452),JPNc.Crop(0,352,0,0))

delogo7=s_ExLogo(yuy2,844,332,44,26,BlurMode=1,LogoMode=2,Amount=255)
art7=ConvertToYV12(delogo7,matrix="PC.709", interlaced=false)
artgrain7=art7.GrainFactory3(g1str=2, g2str=8, g3str=3, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160).Tweak(bright=2.2)
artgrainsh7=StackHorizontal(JPNc.Crop(0,0,-1074,0),artgrain7.Crop(846,0,-1032,0),JPNc.Crop(888,0,0,0))
final7=StackVertical(JPNc.Crop(0,0,0,-472),artgrainsh7.Crop(0,332,0,-452),JPNc.Crop(0,352,0,0))

delogof=s_ExLogo(yuy2,764,390,14,22,BlurMode=0,LogoMode=2,Amount=255)
artf=ConvertToYV12(delogof,matrix="PC.709", interlaced=false)
artgrainf=artf.medianblur(radiusy=2, radiusu=2, radiusv=2, calcborder=false).GrainFactory3(g1str=2, g2str=4, g3str=0, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160)
artgrainshf=StackHorizontal(JPNc.Crop(0,0,-1156,0),artgrainf.Crop(764,0,-1140,0),JPNc.Crop(780,0,0,0))
finalf=StackVertical(JPNc.Crop(0,0,0,-414),artgrainshf.Crop(0,390,0,-392),JPNc.Crop(0,412,0,0))

delogof1=s_ExLogo(yuy2,760,388,20,18,BlurMode=0,LogoMode=2,Amount=255)
artf1=ConvertToYV12(delogof1,matrix="PC.709", interlaced=false)
artgrainf1=artf1.medianblur(radiusy=2, radiusu=2, radiusv=2, calcborder=false).GrainFactory3(g1str=2, g2str=4, g3str=0, g1shrp=38, g2shrp=42, g3shrp=80, g1size=1.0, g2size=0.9, g3size=0.9, g1tex=0, g2tex=0, g3tex=0, temp_avg=0, ontop_grain=0, th1=24, th2=56, th3=128, th4=160).Tweak(bright=2.2)
artgrainshf1=StackHorizontal(JPNc.Crop(0,0,-1160,0),artgrainf1.Crop(760,0,-1146,0),JPNc.Crop(774,0,0,0))
finalf1=StackVertical(JPNc.Crop(0,0,0,-416),artgrainshf1.Crop(0,388,0,-398),JPNc.Crop(0,406,0,0))

deb=f3kdb(JPNc,range=13,Y=42,grainY=32,grainC=0,dynamic_grain=true)


trim(JPNc,0,124759)+trim(final2,124760,124770)+trim(final3,124771,124772)+trim(JPNc,124773,124773)+trim(final4,124774,124774)+trim(final1,124775,124779)+trim(final5,124780,124781)+trim(final6,124782,124782)+trim(final5,124783,124790)+trim(final7,124791,124793)+trim(final5,124794,124796)+trim(JPNc,124797,124914)+trim(finalf,124915,124915)+trim(finalf1,124916,124916)+trim(JPNc,124917,137946)+trim(deb,137947,138035)+trim(JPNc,138036,0)
