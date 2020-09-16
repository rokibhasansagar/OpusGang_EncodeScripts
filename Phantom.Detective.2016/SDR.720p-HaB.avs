 FFVideoSource()
Crop(0, 20, -0, -20)
StackHorizontal(FixColumnBrightness(1,16,190,16,235).FixColumnBrightness(0,16,82,16,235).smoothtweak(saturation=2.5).crop(0,0,2,0),FixColumnBrightness(2,16,237,16,235).crop(2,0,1914,0),FixColumnBrightness(1917,16,236,16,235).smoothtweak(saturation=1.2).crop(1916,0,2,0),FixColumnBrightness(1918,16,188,16,235).
\FixColumnBrightness(1919,16,82,16,235).smoothtweak(saturation=2.5).crop(1918,0,2,0))
StackVertical(fillmargins(0,1,0,0).smoothtweak(saturation=1.5).crop(0,0,0,2),crop(0,2,0,1036),fillmargins(0,0,0,1).smoothtweak(saturation=2.1).crop(0,1038,0,2))
f1=FixRowBrightness(6,16,250,16,235).FixRowBrightness(1039,16,235,16,235).crop(0,2,0,0).AddBorders(0,0,0,2)
f2=FixRowBrightness(6,16,245,16,235).FixRowBrightness(5,16,200,16,235).crop(0,2,0,0).AddBorders(0,0,0,2)
f3=FixRowBrightness(6,16,225,16,235).FixRowBrightness(4,16,300,16,235).crop(0,2,0,0).AddBorders(0,0,0,2)
f4=FixRowBrightness(4,16,235,16,235).crop(0,2,0,0).AddBorders(0,0,0,2)
f5=FixRowBrightness(3,16,250,16,235)
f6=FixRowBrightness(1,16,210,16,235)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalFilter(f5, last, "SceneType", "=", "5")
ConditionalFilter(f6, last, "SceneType", "=", "6")
ConditionalReader("...fix1.txt", "SceneType")
Spline36Resizemod(1280, 694)
f7=StackVertical(crop(0,0,0,2),FixRowBrightness(2,16,79,16,235).smoothtweak(saturation=2).crop(0,2,0,2),crop(0,4,0,688),FixRowBrightness(692,16,140,16,235).smoothtweak(saturation=3).crop(0,692,0,2))
f8=StackVertical(crop(0,0,0,692),FixRowBrightness(692,16,140,16,235).smoothtweak(saturation=3).crop(0,692,0,2))
f9=StackVertical(FixRowBrightness(1,16,115,16,235).smoothtweak(saturation=3).crop(0,0,0,2),crop(0,2,0,690),FixRowBrightness(692,16,140,16,235).smoothtweak(saturation=3).crop(0,692,0,2))
f10=StackVertical(FixRowBrightness(1,16,135,16,235).smoothtweak(saturation=2).crop(0,0,0,2),crop(0,2,0,692))
f11=FixRowBrightness(1,16,225,16,235).FixRowBrightness(0,16,185,16,235)
f12=f3kdb(15,80,80,80,0,0)
f13=f3kdb(15,65,65,65,0,0)
f14=f3kdb(15,100,100,100,0,0)
f15=f3kdb(15,80,80,80,10,10)
ConditionalFilter(f7, last, "SceneType", "=", "1")
ConditionalFilter(f8, last, "SceneType", "=", "2")
ConditionalFilter(f9, last, "SceneType", "=", "3")
ConditionalFilter(f8, last, "SceneType", "=", "4")
ConditionalFilter(f10, last, "SceneType", "=", "5")
ConditionalFilter(f11, last, "SceneType", "=", "6")
ConditionalFilter(f12, last, "SceneType", "=", "7")
ConditionalFilter(f13, last, "SceneType", "=", "8")
ConditionalFilter(f14, last, "SceneType", "=", "9")
ConditionalFilter(f15, last, "SceneType", "=", "10")
ConditionalReader("...fix1.txt", "SceneType")

fix1.txt

type int
default 0

R 0 480 7
R 586 772 7
R 1536 1571 7
R 1950 2002 7
R 2121 2214 7
R 7577 7823 7
R 8259 8320 7
R 8493 8558 7
R 8656 8693 8
R 11973 12349 7
R 17058 17151 7
R 17201 17244 7
R 17276 17302 7
R 17488 17499 7
R 17620 17763 9
R 17824 18055 7
R 18144 18355 7
R 19170 19247 7
R 22822 22867 8
R 25568 25879 7
R 25923 25944 7
R 30703 30828 7
R 30829 30872 7
R 30873 30917 7
R 30918 30950 7
R 31089 31143 7
R 71116 71158 9
R 71210 71499 9
R 77745 77921 9
R 77922 78037 7
R 78038 78078 7
R 78121 78231 7
R 78397 78467 7
R 79905 80465 7
R 89104 89413 9
R 89414 89814 7
R 94633 94807 8
R 101364 101492 7
R 107998 109186 7
R 115992 115993 7
R 120755 120832 9
R 121603 121765 7
R 122612 122835 7
R 124358 124824 8
131443 1
R 131444 131446 2
R 131447 131449 3
R 131450 131453 4
R 131454 131456 5
R 131457 131459 6
R 134821 134917 7
134948 7
R 134984 135099 7
140441 7
R 141836 142040 10
R 142041 142320 7
R 150661 150765 7
R 150783 150831 7
R 150870 151000 7
R 151226 151366 7
R 151453 151525 7
R 151599 151712 7
R 151802 151953 7
R 156540 156569 7
R 160782 160855 8
R 172363 172611 7
R 174852 174967 9
R 175699 175746 9
R 175788 175831 9
R 175832 175893 9
R 175894 175930 9
R 175988 176018 9
