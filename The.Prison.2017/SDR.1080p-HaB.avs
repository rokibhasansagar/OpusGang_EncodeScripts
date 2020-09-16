FFVideoSource()
Crop(0, 138, -0, -138)
f1=StackHorizontal(fillmargins(1,0,0,0).smoothtweak(saturation=3).crop(0,0,2,0),crop(2,0,1918,0))
f2=f3kdb(grainc=0,grainy=10,y=65)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 34306 34449 1
R 116872 116943 2
