FFVideoSource("F:\Workshop\Hellboy\2-video.mkv").Crop(0, 140, -0, -140)
FillBorder(0, 1, 0, 1).bbmod(1, 1, 0, 0, blur=30)
bf = FillBorder(2, 0, 2, 0).bbmod(0, 0, 3, 3, blur=30)
Trim(0, 2001) + bf.Trim(2002, 3996) + Trim(3997, 0)
db00 = f3kdb(range=15, Y=48, Cb=48, Cr=48, grainY=32, grainC=32, dynamic_grain=True, keep_tv_range=True)
db01 = f3kdb(range=15, Y=48, Cb=48, Cr=48, grainY=48, grainC=48, dynamic_grain=True, keep_tv_range=True)
db02 = mt_merge(last, f3kdb(range=15, Y=64, Cb=48, Cr=48, grainY=48, grainC=48, dynamic_grain=True, keep_tv_range=True), mt_binarize(36, true), u=3, v=4)
db03 = f3kdb(range=15, Y=70, Cb=64, Cr=64, grainY=48, grainC=48, dynamic_grain=True, keep_tv_range=True)
GConditionalFilter(db00, last, "ST", "==", "1")
GConditionalFilter(db01, last, "ST", "==", "2")
GConditionalFilter(db02, last, "ST", "==", "3")
GConditionalFilter(db03, last, "ST", "==", "4")
ConditionalReader("filter.txt", "ST", show=False)

----------
Filter.txt

type int
default 0

R 31 499 1
R 3760 3996 1
R 5859 5879 3
R 6533 6568 4
R 6600 6643 2
R 6715 6783 4
R 6809 6830 4
R 6864 6917 4
R 7021 7062 4
R 7222 7255 3
R 7337 7424 3
R 7790 7905 1
R 8974 8985 4
R 10787 10792 2
R 18957 19113 3
R 19165 19193 3
R 20379 20464 4
R 20465 20575 1
R 20724 21139 4
R 21605 21762 4
R 23777 23955 2
R 27184 27486 1
R 30038 30076 3
R 42739 42846 4
R 55646 56456 3
R 67184 67278 4
R 67804 67910 4
R 68264 68318 4
R 84339 84835 3
R 84861 84886 3
R 84932 85072 3
R 85118 85171 3
R 89026 89108 3
R 91127 91161 4
R 96854 97109 3
R 97199 97284 3
R 99224 99251 4
R 99266 99286 4
R 99484 99523 4
R 100881 101072 2
R 101334 101435 4
R 101663 101700 2
R 101762 101836 3
R 101949 101970 4
R 101995 102018 4
R 102194 102236 4
R 102366 102529 4
R 102566 102634 4
R 102746 102820 4
R 104539 104635 3