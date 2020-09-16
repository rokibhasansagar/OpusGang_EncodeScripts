FFVideoSource()
Crop(0,138,0,-138)
FillBorder(0,1,0,0)
StackVertical(crop(0,0,0,16).SmoothTweak(saturation=2).crop(0,0,0,2),crop(0,2,0,0))
f1=f3kdb(10,64,48,48,64,48,keep_tv_range=true,random_algo_ref=2)
f2=FixColumnBrightness(1919,16,180,16,235)
f3=FixColumnBrightness(1917,16,210,16,235)
f4=FixColumnBrightness(1917,16,112,16,235)
f5=FixColumnBrightness(1917,16,132,16,235)
f6=FixColumnBrightness(1917,16,102,16,235)
f7=StackHorizontal(crop(0,0,1918,0),crop(1904,0,0,0).FixColumnBrightness(14,16,140,16,235).FillBorder(0,0,1,0).SmoothTweak(saturation=1.9).crop(14,0,0,0))
f8=StackHorizontal(crop(0,0,1918,0),crop(1904,0,0,0).FixColumnBrightness(14,16,180,16,235).FillBorder(0,0,1,0).SmoothTweak(saturation=1.9).crop(14,0,0,0))
f9=StackHorizontal(crop(0,0,1918,0),crop(1904,0,0,0).FillBorder(0,0,1,0).SmoothTweak(saturation=1.9).crop(14,0,0,0))
f10=FillBorder(0,0,1,0)
f11=StackHorizontal(crop(0,0,1916,0),crop(1904,0,0,0).FixColumnBrightness(12,16,92,16,235).SmoothTweak(saturation=2).crop(12,0,0,0))
f12=StackHorizontal(crop(0,0,1914,0),crop(1904,0,0,0).FixColumnBrightness(10,16,105,16,235).SmoothTweak(saturation=2).crop(10,0,0,0))
f13=StackHorizontal(crop(0,0,1912,0),crop(1904,0,0,0).FixColumnBrightness(8,16,150,16,235).SmoothTweak(saturation=2).crop(8,0,0,0))
f14=FixColumnBrightness(1911,16,150,16,235)
f15=StackHorizontal(crop(0,0,1910,0),crop(0,0,1912,0).BalanceB(0,0,0,2).FixColumnBrightness(1911,16,235,16,16).crop(1910,0,0,0),crop(1912,0,0,0))
f16=FixColumnBrightness(1910,16,170,16,235)
f17=StackHorizontal(crop(0,0,1912,0),crop(0,0,1914,0).FixColumnBrightness(1913,16,150,16,235).BalanceB(0,0,0,2).crop(1912,0,0,0),crop(1914,0,0,0))
f18=StackHorizontal(crop(0,0,1914,0),crop(0,0,1916,0).FixColumnBrightness(1915,16,120,16,235).BalanceB(0,0,0,2).crop(1914,0,0,0),crop(1916,0,0,0))
f19=StackHorizontal(crop(0,0,1918,0),crop(1904,0,0,0).FixColumnBrightness(14,16,180,16,235).Fillborder(0,0,1,0).SmoothTweak(saturation=1.9).crop(14,0,0,0)).BalanceB(0,0,0,2,blur=20)
f20=StackHorizontal(crop(0,0,1918,0),crop(1904,0,0,0).FillBorder(0,0,1,0).SmoothTweak(saturation=1.4).crop(14,0,0,0))
f21=FixColumnBrightness(1916,16,235,16,16)
f22=StackVertical(f1.FixRowBrightness(1,16,235,16,220).FillBorder(0,1,0,0).crop(0,0,0,16).SmoothTweak(saturation=2).crop(0,0,0,2),f1.crop(0,2,0,0))
f23=f1.FixRowBrightness(3,16,235,16,220).FixRowBrightness(2,16,163,16,235)
f24=f1.FixRowBrightness(4,16,235,16,225).FixRowBrightness(3,16,170,16,235)
f25=f1.FixRowBrightness(6,16,235,16,225).FixRowBrightness(5,16,190,16,235).FixRowBrightness(4,16,235,16,16)
f26=f1.FixRowBrightness(7,16,235,16,228).FixRowBrightness(6,16,198,16,235).FixRowBrightness(5,16,235,16,16)
f27=f1.FixRowBrightness(8,16,235,16,230).FixRowBrightness(7,16,205,16,235).FixRowBrightness(6,16,235,16,16)
f28=f1.FixRowBrightness(5,16,235,16,220).FixRowBrightness(4,16,190,16,235)
f29=StackVertical(f1.FixRowBrightness(1,16,150,16,235).FillBorder(0,1,0,0).crop(0,0,0,16).SmoothTweak(saturation=2).crop(0,0,0,2),f1.FixRowBrightness(2,16,235,16,220).crop(0,2,0,0))
f30=FixColumnBrightness(1916,16,233,16,235).FixColumnBrightness(1917,16,230,16,235).FixColumnBrightness(1918,16,228,16,235).FixColumnBrightness(1919,16,223,16,235)
f31=FixRowBrightness(798,16,235,16,223).FixRowBrightness(799,16,182,16,235).FixRowBrightness(800,16,235,16,16)
f32=FixRowBrightness(798,16,233,16,235).FixRowBrightness(799,16,235,16,225).FixRowBrightness(800,16,170,16,235).FixRowBrightness(801,16,235,16,16)
f33=FixRowBrightness(800,16,235,16,220).FixRowBrightness(801,16,155,16,235).FixRowBrightness(802,16,235,16,16)
f34=FixRowBrightness(801,16,235,16,220).FixRowBrightness(802,16,150,16,235).FillBorder(0,0,0,1)
f35=FixRowBrightness(802,16,235,16,220).FixRowBrightness(803,16,140,16,235)
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalFilter(f5, last, "SceneType", "=", "5")
ConditionalFilter(f6, last, "SceneType", "=", "6")
ConditionalFilter(f7, last, "SceneType", "=", "7")
ConditionalFilter(f8, last, "SceneType", "=", "8")
ConditionalFilter(f9, last, "SceneType", "=", "9")
ConditionalFilter(f10, last, "SceneType", "=", "10")
ConditionalFilter(f11, last, "SceneType", "=", "11")
ConditionalFilter(f12, last, "SceneType", "=", "12")
ConditionalFilter(f13, last, "SceneType", "=", "13")
ConditionalFilter(f14, last, "SceneType", "=", "14")
ConditionalFilter(f15, last, "SceneType", "=", "15")
ConditionalFilter(f16, last, "SceneType", "=", "16")
ConditionalFilter(f17, last, "SceneType", "=", "17")
ConditionalFilter(f18, last, "SceneType", "=", "18")
ConditionalFilter(f19, last, "SceneType", "=", "19")
ConditionalFilter(f20, last, "SceneType", "=", "20")
ConditionalFilter(f21, last, "SceneType", "=", "21")
ConditionalFilter(f22, last, "SceneType", "=", "22")
ConditionalFilter(f23, last, "SceneType", "=", "23")
ConditionalFilter(f24, last, "SceneType", "=", "24")
ConditionalFilter(f25, last, "SceneType", "=", "25")
ConditionalFilter(f26, last, "SceneType", "=", "26")
ConditionalFilter(f27, last, "SceneType", "=", "27")
ConditionalFilter(f28, last, "SceneType", "=", "28")
ConditionalFilter(f29, last, "SceneType", "=", "29")
ConditionalFilter(f30, last, "SceneType", "=", "30")
ConditionalFilter(f31, last, "SceneType", "=", "31")
ConditionalFilter(f32, last, "SceneType", "=", "32")
ConditionalFilter(f33, last, "SceneType", "=", "33")
ConditionalFilter(f34, last, "SceneType", "=", "34")
ConditionalFilter(f35, last, "SceneType", "=", "35")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 143 503 1
R 1775 1792 1
R 3472 3564 1
R 3640 3955 1
R 4009 4099 1
R 4138 4385 1
R 4449 4524 1
R 4746 4914 1
R 4960 5223 1
R 5867 5919 1
R 6267 6828 1
R 6909 7602 1
R 7655 7842 1
R 8094 8154 1
R 8318 9056 1
R 9181 9331 1
R 9479 9612 1
R 9646 9776 1
R 9839 10747 1
R 11657 11985 1
R 12215 12264 1
R 12299 12430 1
R 12508 12564 1
R 15166 15263 1
R 18285 18455 1
R 18495 18537 1
R 18701 18751 1
R 18780 18880 1
R 19223 19324 1
R 19480 19554 1
R 20021 20132 1
R 25518 25575 1
R 26437 26483 1
R 26544 26630 1
R 27484 27652 1
R 36348 36372 1
R 37604 37622 1
R 38288 38346 1
R 44599 44753 1
R 46669 46804 1
R 46866 46906 1
R 46976 47016 1
R 47192 47336 1
R 47697 47761 1
R 47803 48033 1
R 49683 49730 1
R 52928 53000 1
R 53307 53350 1
R 53563 53817 1
R 53876 54117 1
R 54324 54420 1
R 62075 62127 1
R 65032 65079 1
R 65283 65348 1
R 65349 65383 1
R 68028 68076 1
R 70492 70645 1
R 72868 72975 1
R 73523 73984 1
R 76036 76073 1
R 76214 76296 1
R 76338 76357 1
R 76524 76617 1
R 76645 76699 1
R 76991 77031 1
R 77470 77529 1
R 77611 77692 1
R 77979 78008 1
R 78185 78233 1
R 78275 78407 1
R 78748 78798 1
R 85169 85215 1
R 85496 85548 1
R 86028 86108 1
R 86188 86295 1
R 86532 86660 1
R 87517 87568 1
R 88105 88461 1
R 92643 92685 1
R 92745 92877 1
R 92949 93004 1
R 105384 105438 1
R 105609 105685 1
R 110239 110286 1
R 112042 112084 1
R 112141 112175 1
R 114908 115305 1
R 115445 115592 1
R 115632 115694 1
R 115802 115941 1
R 116171 116655 1
R 116775 116865 1
R 117817 118014 1
R 119077 119169 1
R 119624 119643 1
R 119995 120063 1
R 120097 120151 1
124034 2
124035 3
124036 4
124037 5
124038 6
124040 7
124041 8
124042 9
124296 10
124297 11
124298 12
124299 13
124300 14
124301 15
R 124302 124304 16
124305 15
124306 17
124307 18
124308 19
R 129409 129410 20
R 129433 129439 20
R 129450 129454 21
R 130941 130981 1
R 132142 132199 1
R 133156 133292 1
R 133799 133914 1
R 134021 134062 1
R 134118 134156 1
R 134314 134340 1
R 135426 135563 1
R 138483 138520 1
R 144459 144503 1
R 144650 144687 1
R 146326 146405 1
R 148032 148128 1
R 148493 148579 1
R 148788 148852 1
R 148957 149241 1
R 149509 149535 1
149536 22
149537 23
149538 24
149539 25
149540 26
R 149541 149544 27
149545 26
149546 25
149547 28
149548 24
149549 23
R 149550 149551 29
R 149552 149554 22
R 150654 151212 1
R 156009 156049 30
R 158756 158816 1
R 159209 159259 1
R 159736 159770 1
R 159860 159868 1
R 159908 159933 1
R 161012 161077 1
R 161321 161376 1
R 161652 161742 1
R 161927 161958 1
R 164834 164897 1
R 179672 179752 1
R 184166 184225 1
R 185212 185288 1
R 185346 185394 1
R 188095 188204 1
R 188372 188573 1
R 188702 188855 1
R 188913 188966 1
R 189330 189744 1
R 193992 194029 1
R 196892 196895 31
R 196896 196899 32
R 196900 196903 33
R 196904 196906 34
R 196907 196910 35
R 205099 205230 1
