FFVideoSource()
Crop(0, 130, -0, -132)
StackVertical(smoothtweak(saturation=1.3).fillmargins(0,1,0,0).BalanceBorders(1,0,0,0).crop(0,0,0,2),crop(0,2,0,816))
Spline36Resizemod(1280, 546)
f1=f3kdb(grainc=0,grainy=0,y=100,range=20)
f2=f3kdb(grainc=0,grainy=0,y=80)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 450 603 1
R 617 726 1
R 179726 179796 2
