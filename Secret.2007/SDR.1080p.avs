ffvideoSource("...")
crop(2, 136, -2, -136)
BalanceBorders(0, 0, 0,10)
fix=StackHorizontal(crop(4,0,4,0),crop(4,0,0,0)).FixColumnBrightness(4,16,145,16,235).FixColumnBrightness(0,16,145,16,235).balanceborders(0,0,14,0)
fix1=StackHorizontal(crop(0,0,4,0),crop(2,0,-2,0)).fillmargins(0,0,2,0)
fix2=StackHorizontal(crop(4,0,4,0),crop(4,0,0,0))
ConditionalFilter(last, fix, "SceneType", "<", "1")
ConditionalFilter(last, fix1, "SceneType", "<", "2")
ConditionalFilter(last, fix2, "SceneType", "<", "3")
ConditionalReader("...fix.txt", "SceneType")

fix.txt:
type int
default 0

R 8292 9135 1
R 53320 53526 2
R 53620 53802 2
R 52500 52571 3
