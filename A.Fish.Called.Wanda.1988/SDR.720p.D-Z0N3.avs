FFVideoSource()
Crop(0, 20, -0, -20)
StackVertical(crop(0,0,0,6).FillBorder(0,1,0,0).Tweak(sat=1.7,dither=true).crop(0,0,0,2),crop(0,2,0,1036),crop(0,1034,0,0).FillBorder(0,0,0,1).Tweak(sat=1.7,dither=true).crop(0,4,0,0))
z_ConvertFormat(1280,692,src_top=1,src_height=-1,resample_filter="spline36",dither_type="ordered",cpu_type="skl-x")
