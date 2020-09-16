LWLibavVideoSource()
Crop(0, 140, -0, -140)
BalanceB(0,0,0,1,thresh=40)
StackHorizontal(crop(0,0,1918,0),crop(1904,0,0,0).SmoothTweak(saturation=1.15,usemt=8).crop(14,0,0,0))
