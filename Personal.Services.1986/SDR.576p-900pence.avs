# This is a really problematic source. The video is interlaced, but when separating them the field themselves appear blended. After having tried all options I possibly could, it seems to me that the video was first telecined by adding blended frames to the progressive video; this is not the standard way to telecine, but apparently it happens frequent enough that someone wrote a plugin, FixBlendIVTC, to handle it (its functionality was later incorporated into srestore, which is what I've used). Afterwards, the resulting progressive video was interlaced with normal techniques. You can see the script I've used below.


#source
Mpeg2Source("VTS_01_1.d2v")

#deinterlacing
AssumeTFF.QTGMC()
 
#IVTC and decimation
srestore() 

#crop, duh!
Crop(2, 2, -2, -2)

#fix one dirty line
bbmod(0, 0, 1, 0)



# One last note: by default, srestore guesses the output framerate used for decimation, which in this case turned out to be 23.976. However, as averroista had employed a framerate of exactly 24 for his encode, I wanted to doublecheck. To avoid errors due to values cached by srestore (which seems to be quite frequent when using Avspmod's preview) I've made two complete encode: one using the script above, and another forcing 24fps via


# srestore(frate=24) 


# I've then interleaved the two resulting encodes, and looked for instances when they went out of sync — that is, for the occasional frame from the 24fps encode that ws missing from the 23.976fps one. Out of 12 such instances, 11 turned out to be duplicate frames in the 24fps encode, while in another I couldn't tell for sure whether the differences were due to minor compression artifacts (since it was a scene with two people standing still). Thus, it seems reasonable to me that 23.976fps be the correct framerate.
