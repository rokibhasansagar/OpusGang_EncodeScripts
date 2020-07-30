FFVideoSource("C:\Users\xxxxxx\Desktop\tomboy_enc\source\video.h264")

SelectRangeEvery(2000, 50, 10000)

Crop(2, 20, -0, -20)
FillBorders(0, 1, 0, 1)

SmoothLevels(input_low=21,output_low=16,input_high=230,output_high=235)

z_Spline36Resize(1280, 694, 0, 1, 0, -1, dither="error_diffusion")

mt_merge(f3kdb(dither_algo=3,y=70,cb=70,cr=70,grainy=20,grainc=20), mt_binarize(80, true), luma=true)
