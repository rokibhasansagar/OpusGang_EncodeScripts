FFVideoSource()
Crop(234, 0, -234, -0)
f1=FixColumnBrightness(4,16,240,16,235)
f2=StackHorizontal(fillmargins(1,0,0,0).BalanceBorders(0,0,1,0).tweak(sat=1.7).crop(0,0,2,0),crop(2,0,0,0))
f3=StackHorizontal(crop(2,0,2,0),crop(2,0,0,0))
f4=StackHorizontal(fillmargins(1,0,0,0).BalanceBorders(0,0,1,0).tweak(sat=1.7).crop(0,0,2,0),FixRowBrightness(2,16,238,16,235).FixRowBrightness(1,16,233,16,235).crop(2,0,0,0))
f5=FixColumnBrightness(1447,16,238,16,235).FixColumnBrightness(1449,16,250,16,235).FixColumnBrightness(4,16,240,16,235).FixColumnBrightness(2,16,250,16,235)
f6=FixColumnBrightness(1449,16,250,16,235).FixColumnBrightness(4,16,240,16,235).FixColumnBrightness(2,16,250,16,235)
f7=FixRowBrightness(2,16,238,16,235).FixRowBrightness(1,16,233,16,235).FixColumnBrightness(1449,16,250,16,235)
f8=StackHorizontal(FixColumnBrightness(1,16,240,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,1,0).tweak(sat=1.7).crop(0,0,2,0),FixColumnBrightness(2,16,230,16,235).fixColumnBrightness(1450,16,250,16,235).fillmargins(0,0,1,0).BalanceBorders(0,0,0,1).crop(2,0,0,0))
f9=stackHorizontal(FixColumnBrightness(1,16,225,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,1,0).tweak(sat=1.7).crop(0,0,2,0),FixColumnBrightness(4,16,240,16,235).FixColumnBrightness(2,16,250,16,235).FixColumnBrightness(1447,16,240,16,235).FixColumnBrightness(1449,16,252,16,235).FixColumnBrightness(1450,16,215,16,235).fillmargins(0,0,1,0).BalanceBorders(0,0,0,1).crop(2,0,0,0))
f10=f8.FixRowBrightness(1078,16,215,16,235).FixrowBrightness(1079,16,180,16,235)
f11=f8.FixRowBrightness(7,16,233,16,235).FixRowBrightness(4,16,230,16,235).FixrowBrightness(2,16,218,16,235)
ConditionalFilter(last, f1, "SceneType", "<", "1")
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalFilter(last, f6, "SceneType", "<", "6")
ConditionalFilter(last, f7, "SceneType", "<", "7")
ConditionalFilter(last, f8, "SceneType", "<", "8")
ConditionalFilter(last, f9, "SceneType", "<", "9")
ConditionalFilter(last, f10, "SceneType", "<", "10")
ConditionalFilter(last, f11, "SceneType", "<", "11")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 1056 3375 1
R 3376 15410 2
R 15411 15411 3
R 15412 15418 2
R 15419 16398 4
R 16399 18836 5
R 18837 20098 4
R 20099 20813 6
R 20814 20814 7
R 20815 21748 4
R 21749 25606 2
R 25607 32200 4
R 32201 36507 8
R 36508 36597 9
R 36598 37353 8
R 36598 37429 9
R 37430 57862 8
R 57863 57866 10
R 57867 61857 8
R 61858 68552 2
R 68553 70247 4
R 70248 74801 2
R 74802 92056 4
R 92057 93774 2
R 93775 96915 4
R 96916 98274 2
R 98275 98449 4
R 98450 99386 2
R 99387 99577 4
R 99578 103596 2
R 103597 104415 4
R 104416 104516 2
R 104517 141006 4
R 141007 142089 8
R 142090 144124 4
R 144125 144972 11
R 144974 145286 4
R 145286 145351 11
R 145353 145407 4
R 145408 145535 11
R 145537 159087 4
