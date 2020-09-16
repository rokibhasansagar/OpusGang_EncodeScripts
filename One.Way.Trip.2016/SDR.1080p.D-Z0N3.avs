FFVideoSource()
Crop(2, 136, -2, -136) #1px black line + 1px fixable line but duplicated are cropped on both sides
FixRowBrightness(0,16,132,16,255).FixRowBrightness(807,16,138,16,255)
StackHorizontal(SmoothTweak(saturation=1.2).crop(0,0,2,0),crop(2,0,1912,0),SmoothTweak(saturation=1.2).crop(1914,0,2,0))
StackVertical(SmoothTweak(saturation=1.2).crop(0,0,0,2),crop(0,2,0,804),SmoothTweak(saturation=1.2).crop(0,806,0,2))
Trim(0,188)+Trim(f3kdb(15,65,64,64,0,0),189,538)+Trim(539,0)
