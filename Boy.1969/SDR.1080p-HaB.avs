FFVideoSource().Crop(0, 140, -0, -140)
f1=FixColumnBrightness(0,16,117,16,235).FixColumnBrightness(1919,16,117,16,235)
f2=StackVertical(f1.crop(0,0,0,64),FixColumnBrightness(0,16,117,16,235).BalanceBorders(0,0,0,1,blur=4).crop(0,64,0,88),f1.crop(0,152,0,648))
f3=FixColumnBrightness(1919,16,117,16,235).BalanceBorders(0,0,2,0,blur=30)
f4=FixColumnBrightness(1919,16,117,16,235).BalanceBorders(0,0,2,0)
f5=FixRowBrightness(1,16,238,16,235).FixRowBrightness(0,16,205,16,235).FixColumnBrightness(2,16,245,16,235).FixColumnBrightness(1,16,210,16,235).FixColumnBrightness(0,16,75,16,235).FixRowBrightness(799,16,242,16,235).FixColumnBrightness(1918,16,250,16,235).FixColumnBrightness(1919,16,170,16,235).BalanceBorders(2,1,1,0)
f6=FixRowBrightness(1,16,248,16,235).FixColumnBrightness(0,16,150,16,235).FixRowBrightness(799,16,242,16,235).FixColumnBrightness(1919,16,150,16,235).BalanceBorders(2,1,0,0).BalanceBorders(0,0,2,2,blur=100)
f7=FixRowBrightness(1,16,248,16,235).FixColumnBrightness(0,16,150,16,235).FixRowBrightness(797,16,230,16,235).FixRowBrightness(798,16,245,16,235).FixColumnBrightness(1919,16,150,16,235).BalanceBorders(2,1,0,0).BalanceBorders(0,0,2,2,blur=100)
f8=FixRowBrightness(1,16,248,16,235).FixColumnBrightness(0,16,150,16,235).FixRowBrightness(799,16,242,16,235).FixColumnBrightness(1917,16,245,16,235).FixColumnBrightness(1919,16,100,16,235).BalanceBorders(2,1,0,0).BalanceBorders(0,0,2,2,blur=100)
f9=FixRowBrightness(1,16,230,16,235).FixRowBrightness(0,16,217,16,235).FixColumnBrightness(1,16,205,16,235).FixColumnBrightness(0,16,130,16,235).FixRowBrightness(799,16,220,16,235).FixcolumnBrightness(1917,16,230,16,235).FixColumnBrightness(1918,16,200,16,235).FixColumnBrightness(1919,16,130,16,235).
\BalanceBorders(1,0,2,2).BalanceBorders(0,1,0,0,blur=20)
f10=StackVertical(FixColumnBrightness(0,16,117,16,235).FixRowBrightness(0,16,250,16,235).FixColumnBrightness(1918,16,245,16,235).fillmargins(0,0,1,0).BalanceBorders(1,1,2,0).crop(0,0,0,26),f1.FixRowBrightness(0,16,250,16,235).BalanceBorders(1,1,2,2).crop(0,26,0,774))
f11=f1.FixRowBrightness(0,16,250,16,235).BalanceBorders(1,1,2,2)
f12=BalanceBorders(0,0,2,1)
f13=StackHorizontal(BalanceBorders(1,0,0,0,blur=15).BalanceBorders(0,0,2,0).crop(0,0,388,0),FixColumnBrightness(1919,16,117,16,235).crop(388,0,1532,0))
f14=FixRowBrightness(1,16,248,16,235).FixRowBrightness(799,16,242,16,235).FixColumnBrightness(1919,16,150,16,235).FixColumnBrightness(2,16,245,16,235).FixColumnBrightness(1,16,200,16,235).FixColumnBrightness(0,16,65,16,235).fillmargins(1,0,0,0).BalanceBorders(2,1,2,0).BalanceBorders(0,0,0,2,blur=100)
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
ConditionalFilter(last, f12, "SceneType", "<", "12")
ConditionalFilter(last, f13, "SceneType", "<", "13")
ConditionalFilter(last, f14, "SceneType", "<", "14")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 0 41851 1
41852 2
R 41853 57066 1
R 57067 57408 3
R 57409 57805 4
R 57806 71790 1
R 71791 71793 5
R 71794 71799 14
71800 6
71801 7
R 71802 71804 6
71805 7
71806 6
71807 7
R 71808 71818 6
71819 7
R 71820 71827 6
71828 7
R 71829 71836 6
71837 7
R 71838 71843 6
71844 8
R 71845 71848 6
71849 8
R 71850 71860 6
71861 7
R 71862 71864 6
71865 8
R 71866 71880 6
71881 8
R 71882 71888 6
71889 7
71890 6
71891 7
R 71892 71899 6
R 71900 71901 8
R 71902 71928 6
71929 8
R 71930 71937 6
71938 8
R 71939 71944 6
71945 8
R 71946 71956 6
71957 7
R 71958 71959 6
R 71960 71962 8
R 71963 71970 6
71971 8
71972 7
R 71973 71986 6
R 71987 71988 8
R 71989 72008 6
72009 8
R 72010 72019 6
72020 7
72021 6
72022 7
R 72023 72024 6
72025 8
72026 7
R 72027 72033 6
72034 7
R 72035 72052 6
72053 7
R 72054 72056 6
72057 8
72058 6
72059 7
R 72060 72066 6
R 72067 72068 8
72069 6
72070 8
72071 6
R 72072 72074 8
72075 6
R 72076 72078 7
R 72079 72083 6
72084 7
R 72085 72088 6
72089 8
R 72090 72099 6
72100 7
R 72101 72136 6
72137 8
R 72138 72139 6
72140 7
R 72141 72160 6
R 72161 72165 7
R 72166 72178 6
72179 8
R 72180 72200 6
72201 8
R 72202 72232 6
72233 7
R 72234 72234 6
72235 7
R 72236 72238 6
72239 7
R 72240 72254 6
R 72255 72256 7
R 72257 72296 6
R 72297 72298 8
72299 6
R 72300 72301 7
R 72302 72308 6
72309 7
R 72310 72321 6
72322 8
72323 6
72324 8
R 72325 72328 6
72329 8
R 72330 72344 6
72345 8
R 72346 72355 6
72356 7
R 72357 72370 6
72371 8
R 72372 72375 6
R 72376 110846 1
R 110847 111051 9
R 111052 112127 1
112128 10
R 112129 112218 11
112219 12
R 112220 116848 1
R 116849 117041 11
R 117042 119093 1
R 119094 119486 13
R 119487 139525 1
