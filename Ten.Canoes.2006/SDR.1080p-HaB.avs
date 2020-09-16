FFVideoSource()
f1=StackVertical(StackHorizontal(FixColumnBrightness(1,16,250,16,235).FixColumnBrightness(0,16,210,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,2,0).crop(0,0,2,0),crop(2,0,0,0)).crop(0,0,0,252),crop(0,252,0,0))
f2=StackVertical(StackHorizontal(FixColumnBrightness(1,16,250,16,235).FixColumnBrightness(0,16,210,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,2,0).crop(0,0,2,0),crop(2,0,0,0)).crop(0,0,0,652),crop(0,652,0,0))
f3=StackVertical(StackHorizontal(FixColumnBrightness(1,16,250,16,235).FixColumnBrightness(0,16,210,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,2,0).crop(0,0,2,0),crop(2,0,0,0)).crop(0,0,0,326),crop(0,326,0,0))
f4=StackVertical(StackHorizontal(FixColumnBrightness(1,16,250,16,235).FixColumnBrightness(0,16,210,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,2,0).crop(0,0,2,0),crop(2,0,0,0)).crop(0,0,0,308),crop(0,308,0,0))
f5=FixColumnBrightness(2,16,250,16,235).FixColumnBrightness(1,16,245,16,235).FixColumnBrightness(0,16,150,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,2,0)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 69146 71080 1
R 72942 73563 2
R 76901 77351 3
R 84620 85132 4
R 98935 99550 5
