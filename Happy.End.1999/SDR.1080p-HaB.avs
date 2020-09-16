FFVideoSource()
FixColumnBrightness(0,16,143,16,235)
FixColumnBrightness(1919,16,120,16,235)
BalanceBorders(0,0,1,1)
f0=crop(0,4,0,0).AddBorders(0,0,0,4)
f1=crop(0,6,0,0).AddBorders(0,0,0,6)
f2=crop(0,8,0,0).AddBorders(0,0,0,8)
f3=crop(0,2,0,0).AddBorders(0,0,0,2)
ConditionalFilter(last, f0, "SceneType", "<", "1")
ConditionalFilter(last, f1, "SceneType", "<", "2")
ConditionalFilter(last, f2, "SceneType", "<", "3")
ConditionalFilter(last, f3, "SceneType", "<", "4")
ConditionalReader("...happy.txt", "SceneType")

happy.txt

type int
default 0

R 900 3548 1
R 16613 16726 2
R 24668 25391 2
R 35442 35445 3
R 40859 42224 2
R 50748 51355 1
R 62512 62989 4
R 62990 62993 3
R 69550 70274 2
R 92467 93479 2
R 93480 93482 3
R 104676 105737 2
R 108419 108748 1
R 119060 119692 1
R 124622 126646 2
R 134159 135608 2
R 137186 137378 1
