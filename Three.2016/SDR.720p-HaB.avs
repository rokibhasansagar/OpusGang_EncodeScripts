 FFVideoSource()
Crop(0, 132, -0, -130)
f1=FixColumnBrightness(1918,16,220,16,235).FixColumnBrightness(1919,16,110,16,235).FixColumnBrightness(1,16,220,16,235).FixColumnBrightness(0,16,115,16,235).FixRowBrightness(815,16,195,16,235).FixRowBrightness(0,16,185,16,235)
f2=FixColumnBrightness(1919,16,168,16,235).FixColumnBrightness(0,16,170,16,235)
f3=fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).BalanceBorders(1,0,0,0)
f4=FixRowBrightness(814,16,243,16,235)
f5=FixRowBrightness(812,16,230,16,235).FixRowBrightness(813,16,230,16,235).FixRowBrightness(814,16,223,16,235).FixRowBrightness(815,16,242,16,235).FixRowBrightness(3,16,232,16,235).BalanceBorders(1,0,0,0)
f6=FixRowBrightness(814,16,240,16,235).FixRowBrightness(815,16,250,16,235).FixRowBrightness(1,16,240,16,235).BalanceBorders(1,0,0,0)
f7=FixRowBrightness(4,16,230,16,235).FixRowBrightness(2,16,220,16,235).FixRowBrightness(1,16,245,16,235).FixRowBrightness(812,16,230,16,235).FixRowBrightness(813,16,233,16,235).FixRowBrightness(814,16,220,16,235).FixRowBrightness(816,16,210,16,235).fillmargins(0,0,0,1).BalanceBorders(1,2,0,0,blur=80)
f8=f3.FixRowBrightness(2,16,243,16,235)
f9=f3.FixRowBrightness(2,16,231,16,235)
f10=FixRowBrightness(2,16,240,16,235).FixRowBrightness(1,16,280,16,235).fillmargins(0,1,0,0).BalanceBorders(2,0,0,0).FixRowBrightness(814,16,260,16,235).FixRowBrightness(815,16,165,16,235)
f11=FixRowBrightness(815,16,200,16,235).BalanceBorders(1,0,0,0)
f12=FixRowBrightness(4,16,230,16,235).FixRowBrightness(2,16,220,16,235).FixRowBrightness(1,16,245,16,235).FixRowBrightness(812,16,240,16,235).FixRowBrightness(813,16,227,16,235).FixRowBrightness(814,16,238,16,235).FixRowBrightness(815,16,205,16,235).FixRowBrightness(816,16,265,16,235).fillmargins(0,0,0,1).BalanceBorders(1,2,0,0,blur=80)
f13=StackVertical(FixRowBrightness(1,16,282,16,235).fillmargins(0,1,0,0).BalanceBorders(1,0,0,0).smoothtweak(saturation=1.7).crop(0,0,0,2),FixRowBrightness(2,16,241,16,235).FixRowBrightness(815,16,245,16,235).fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).crop(0,2,0,0))
f14=FixRowBrightness(3,16,224,16,235).FixRowBrightness(1,16,198,16,235).FixRowBrightness(0,16,245,16,235).FixRowBrightness(813,16,228,16,235).FixRowBrightness(815,16,210,16,235).FixRowBrightness(816,16,260,16,235).fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).BalanceBorders(1,0,0,0)
f15=StackVertical(FixRowBrightness(1,16,285,16,235).fillmargins(0,1,0,0).BalanceBorders(1,0,0,0).smoothtweak(saturation=1.5).crop(0,0,0,2),fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).crop(0,2,0,0))
f16=FixColumnBrightness(1916,16,228,16,235).FixColumnBrightness(1917,16,170,16,235).FixColumnBrightness(1918,16,145,16,235).fillmargins(1,0,1,1).BalanceBorders(1,0,2,1).BalanceBorders(0,2,0,0,blur=80)
f17=f11.FixRowBrightness(1,16,238,16,235)
f18=FixRowBrightness(816,16,178,16,235).fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).BalanceBorders(1,0,0,0)
f19=FixRowBrightness(1,16,238,16,235).BalanceBorders(1,0,0,0)
f20=FixColumnBrightness(1917,16,245,16,235).FixRowBrightness(815,16,212,16,235).BalanceBorders(1,0,0,0)
f21=FixRowBrightness(814,16,240,16,235).FixRowBrightness(815,16,255,16,235).BalanceBorders(1,0,0,0)
f22=FixRowBrightness(815,16,250,16,235).fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).BalanceBorders(1,0,0,0)
f23=FixRowBrightness(814,16,240,16,235).FixRowBrightness(815,16,250,16,235).FixRowBrightness(2,16,245,16,235).FixRowBrightness(1,16,290,16,235)
f24=FixRowBrightness(1,16,225,16,235).FixRowBrightness(0,16,220,16,235).fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80)
f25=FixRowBrightness(3,16,233,16,235).FixRowBrightness(1,16,230,16,235).fillmargins(0,0,0,1).BalanceBorders(0,2,0,0,blur=80).BalanceBorders(1,0,0,0)
f26=StackVertical(FixRowBrightness(1,16,218,16,235).fillmargins(0,1,0,0).smoothtweak(saturation=1.7).crop(0,0,0,2),FixRowBrightness(815,16,200,16,235).crop(0,2,0,0))
f27=FixRowBrightness(815,16,200,16,235).FixRowBrightness(1,16,238,16,235).fillmargins(0,0,1,0).BalanceBorders(1,0,0,1)
f28=BalanceBorders(1,0,0,0)
f29=StackVertical(FixRowBrightness(2,16,238,16,235).FixRowBrightness(1,16,260,16,235).FixRowBrightness(0,16,250,16,235).FixRowBrightness(813,16,242,16,235).FixRowBrightness(814,16,260,16,235).FixRowBrightness(815,16,270,16,235).crop(0,0,0,816).BalanceBorders(2,2,0,0),crop(0,816,0,0))
f30=FixRowBrightness(1,16,245,16,235).FixRowBrightness(0,16,250,16,235).BalanceBorders(2,0,0,0).FixRowBrightness(816,16,238,16,235).FixRowBrightness(814,16,245,16,235).FixRowBrightness(815,16,245,16,235)
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
ConditionalFilter(last, f15, "SceneType", "<", "15")
ConditionalFilter(last, f16, "SceneType", "<", "16")
ConditionalFilter(last, f17, "SceneType", "<", "17")
ConditionalFilter(last, f18, "SceneType", "<", "18")
ConditionalFilter(last, f19, "SceneType", "<", "19")
ConditionalFilter(last, f20, "SceneType", "<", "20")
ConditionalFilter(last, f21, "SceneType", "<", "21")
ConditionalFilter(last, f22, "SceneType", "<", "22")
ConditionalFilter(last, f23, "SceneType", "<", "23")
ConditionalFilter(last, f24, "SceneType", "<", "24")
ConditionalFilter(last, f25, "SceneType", "<", "25")
ConditionalFilter(last, f26, "SceneType", "<", "26")
ConditionalFilter(last, f27, "SceneType", "<", "27")
ConditionalFilter(last, f28, "SceneType", "<", "28")
ConditionalFilter(last, f29, "SceneType", "<", "29")
ConditionalFilter(last, f30, "SceneType", "<", "30")
ConditionalReader("...fix.txt", "SceneType")
Spline36Resizemod(1280, 546)

fix.txt

type int
default 0

R 189 287 1
R 288 867 2
R 868 4487 3
R 4488 4562 4
R 4563 8330 3
R 8331 8400 8
R 8401 11091 3
R 11092 12205 5
R 12206 14512 3
R 14513 14612 6
R 14613 14756 3
R 14755 14908 9
R 14909 15008 3
R 15009 15254 9
R 15255 15409 3
R 15410 15577 9
R 15578 15830 3
R 15831 15956 6
R 15957 19187 3
R 19394 23200 3
R 23201 23303 7
R 23304 23898 3
R 23899 24048 9
R 24049 24132 3
R 24133 24189 7
R 24190 32013 3
R 32014 32250 10
R 32251 36188 3
R 36345 36899 3
R 37031 38709 3
R 38710 38743 11
R 38744 40588 3
R 40654 41031 3
R 41116 41336 3
R 41480 41566 3
R 41692 51048 3
R 51049 51245 12
R 51246 51377 3
R 51378 51541 13
R 51542 51998 14
R 51999 52397 15
R 52398 53225 3
R 53448 53935 3
R 54217 68960 3
R 68961 69106 11
R 69107 75507 3
R 75508 75677 16
R 75678 84108 3
R 84109 84144 11
R 84145 89282 3
R 89283 89366 11
R 89367 89470 3
R 89471 89640 11
R 89641 89864 3
R 89865 89902 11
R 89903 90415 3
R 90416 90573 17
R 90574 104348 3
R 104500 104568 18
R 104569 107738 3
R 107806 107936 3
R 108001 108401 3
R 108402 108446 17
R 108447 108681 19
R 108682 108905 17
R 108906 109309 3
R 109310 109347 17
R 109348 109361 3
R 109362 109378 19
R 109379 109392 3
R 109393 109425 20
R 109426 109519 11
R 109520 109641 3
R 109642 109758 11
R 109759 110090 3
R 110091 110145 21
R 110146 110211 22
R 110212 110278 23
R 110279 110445 3
R 110446 110524 24
R 110525 110593 3
R 110594 110654 25
R 110655 110717 11
R 110724 110834 3
R 110835 111219 11
R 111220 111392 3
R 111393 111660 26
R 111661 111902 3
R 111903 112027 27
R 112028 112109 3
R 112110 112170 11
R 112171 112275 3
R 112276 112316 11
R 112317 112422 3
R 112423 112479 11
R 112480 112612 3
R 112613 112655 11
R 112613 112752 3
R 112753 112985 11
R 112986 113023 3
R 113024 113158 11
R 113159 113227 3
R 113228 113295 28
R 113296 113348 29
R 113349 113446 3
R 113447 113517 30
R 113518 113551 11
R 113552 113624 30
R 113625 113699 11
R 113700 113768 3
R 113769 113975 11
R 113976 113984 3
R 113985 114001 11
R 114002 114038 3
R 114039 114178 11
R 114179 114222 30
R 114223 114438 11
R 114439 114742 3
R 114743 114793 11
R 114794 115014 3
R 115015 115058 11
R 115059 115095 3
R 115096 115226 11
R 115227 115379 3
R 115380 115450 11
R 115451 115527 3
R 115528 115589 11
R 115590 115660 3
R 115661 115708 11
R 115709 115859 3
R 115859 115880 11
R 115881 115900 3
R 115901 115962 11
R 115963 116103 3
R 116104 116169 29
R 116170 116247 3
R 116248 116643 11
R 116644 117638 3
R 117639 117750 11
R 117751 126764 3
