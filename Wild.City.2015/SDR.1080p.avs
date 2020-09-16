ffvideoSource()
crop(0, 132, 0, -132)
fix=FixColumnBrightness(1918,16,230,16,235).FixColumnBrightness(1919,16,75,16,235).FixColumnBrightness(1,16,200,16,235).FixColumnBrightness(0,16,80,16,235).balanceborders(0,0,1,1)
ConditionalFilter(last, fix, "SceneType", "<", "1")
ConditionalReader("...fix.txt", "SceneType")

fix.txt:
type int
default 0

R 0 7546 1
R 7786 8154 1
R 8801 22365 1
