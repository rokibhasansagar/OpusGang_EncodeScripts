FFVideoSource("Source.mkv")
crop(0, 138, 0, -138)

FillBorder(0,0,0,1).BalanceBorders(0,2,0,0,128,10)

filter1=f3kdb(dither_algo=3, grainY=45, grainC=45, Y=64, Cb=64, Cr=64, dynamic_grain=true)
filter2=FixColumnBrightnessProtect2(2,22,10).FixColumnBrightnessProtect2(1917,22,10).FillBorder(2,0,2,0).BalanceBorders(0,0,2,2,128,10).f3kdb(dither_algo=3, grainY=45, grainC=45, Y=64, Cb=64, Cr=64, dynamic_grain=true)
filter3=FixColumnBrightnessProtect2(0,50,10).FixColumnBrightnessProtect2(1,22,10).FixColumnBrightnessProtect2(2,3,10).BalanceBorders(0,0,3,0,128,5)
filter4=FixRowBrightnessProtect2(0,-7,10).FixRowBrightnessProtect2(1,4,10).FixRowBrightnessProtect2(2,-1,10)
filter5=FixColumnBrightnessProtect2(0,20,10).BalanceBorders(0,0,2,0,128,5)
filter6=FixRowBrightnessProtect2(0,-7,10).FixRowBrightnessProtect2(1,4,10).FixRowBrightnessProtect2(2,-1,10).FixRowBrightnessProtect2(801,1,10).FixRowBrightnessProtect2(800,-1,10)
filter7=FixColumnBrightnessProtect2(1919,50,10).FixColumnBrightnessProtect2(1918,22,10).FixColumnBrightnessProtect2(1917,3,10).BalanceBorders(0,0,0,3,128,5)
filter8=BalanceBorders(0,3,0,0,128,10)

ConditionalFilter(last, filter1, "SceneType", "<", "1")
ConditionalFilter(last, filter2, "SceneType", "<", "2")
ConditionalFilter(last, filter3, "SceneType", "<", "3")
ConditionalFilter(last, filter4, "SceneType", "<", "4")
ConditionalFilter(last, filter5, "SceneType", "<", "5")
ConditionalFilter(last, filter6, "SceneType", "<", "6")
ConditionalFilter(last, filter7, "SceneType", "<", "7")
ConditionalFilter(last, filter8, "SceneType", "<", "8")
ConditionalReader("fix.txt", "SceneType")

# fix.txt
type int
default 0

R 72 291 1
R 325 680 2
R 1739 2086 3
R 3252 3298 7
R 3982 4967 1
R 6973 7052 4
R 7771 7957 1
R 8151 8315 1
R 8409 8717 1
R 8934 9107 1
R 9230 9371 1
R 13187 13220 1
R 15575 15623 1
R 24773 25128 1
R 30825 30918 1
R 33498 33574 5
R 42417 42465 1
R 44023 44183 1
R 44412 44508 1
R 44558 44639 1
R 44776 44844 1
R 44942 45024 1
R 45227 45379 1
R 45522 45587 1
R 45684 45770 1
R 45953 46041 1
R 46202 46500 1
R 46575 46661 1
R 46792 46895 1
R 47308 47634 1
R 48026 48766 1
R 49365 49425 1
R 49575 49733 1
R 49982 50215 1
R 64616 64660 1
R 64952 65121 1
R 65558 65766 1
R 71001 71142 1
R 72453 72524 1
R 78842 79099 1
R 84030 84923 1
R 89704 89817 1
R 90459 90693 1
R 94256 94325 1
R 95002 95114 1
R 99061 99190 1
R 99414 99495 1
R 99552 99598 8
R 100586 103117 1
R 106729 106789 1
R 109462 109936 1
R 110102 112035 1
R 112490 112816 1
R 116442 116716 1
R 117721 117828 1
R 118768 118814 1
R 119972 120016 1
R 120102 120268 1
R 121708 121735 1
R 126351 126540 1
R 129336 129877 6
R 128801 129105 1
R 132204 132286 1