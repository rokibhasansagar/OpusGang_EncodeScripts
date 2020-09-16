 FFVideoSource()
Crop(0, 20, -0, -22)
StackVertical(FixColumnBrightness(1918,16,230,16,235).FixColumnBrightness(1919,16,240,16,235).fillmargins(0,1,0,0).BalanceBorders(1,0,0,0).smoothtweak(saturation=1.4).crop(0,0,0,2),FixColumnBrightness(1918,16,230,16,235).FixColumnBrightness(1919,16,240,16,235).crop(0,2,0,1036))
f2=FixColumnBrightness(2,16,225,16,235).fillmargins(1,0,0,0).BalanceBorders(0,0,1,0)
f3=FixColumnBrightness(3,16,220,16,235)
f4=FixColumnBrightness(4,16,225,16,235).FixColumnBrightness(3,16,245,16,235).FixColumnBrightness(2,16,210,16,235)
f5=FixColumnBrightness(1916,16,220,16,235).FixColumnBrightness(1917,16,240,16,235)
ConditionalFilter(last, f2, "SceneType", "<", "2")
ConditionalFilter(last, f3, "SceneType", "<", "3")
ConditionalFilter(last, f4, "SceneType", "<", "4")
ConditionalFilter(last, f5, "SceneType", "<", "5")
ConditionalReader("...fix.txt", "SceneType")

fix.txt

type int
default 0

R 2972 2973 2
R 2979 2986 2
R 2987 2987 3
R 2988 2988 2
R 2989 2990 3
R 2991 2991 2
R 2992 2995 3
R 2996 2998 2
R 3000 3000 3
R 3001 3002 2
R 3003 3003 3
R 3004 3004 2
R 3005 3006 3
R 3007 3010 2
R 3012 3013 2
R 3014 3014 3
R 3015 3016 2
R 3017 3019 3
R 3020 3020 4
R 3021 3027 3
R 3028 3028 2
R 3029 3029 4
R 3030 3039 3
R 3040 3040 4
R 3041 3046 3
R 3047 3047 2
R 3048 3048 3
R 3049 3054 2
R 135622 135841 5
