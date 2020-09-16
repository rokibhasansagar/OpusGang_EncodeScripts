FFVideoSource()
Crop(0, 20, -0, -20)
s=last
extracty()
crop(0,1,0,-1)
y=last
s
extractu()
ConvertBits(16)
nnedi3(0,true,nsize=4,nns=2,qual=2,pscrn=0,threads=1).z_convertformat(960,519,src_left=0,src_top=1.5,src_width=960,src_height=1038,resample_filter="lanczos",filter_param_a=6,dither_type="error_diffusion",cpu_type="skl-x")
ConvertBits(8,dither=1)
u=last
s
extractv()
ConvertBits(16)
nnedi3(0,true,nsize=4,nns=2,qual=2,pscrn=0,threads=1).z_convertformat(960,519,src_left=0,src_top=1.5,src_width=960,src_height=1038,resample_filter="lanczos",filter_param_a=6,dither_type="error_diffusion",cpu_type="skl-x")
ConvertBits(8,dither=1)
CombinePlanes(y,u,last,planes="YUV",pixel_type="YV12")

f1=f3kdb(15,64,48,48,64,48,dither_algo=2,keep_tv_range=true,random_algo_ref=2)
f2=StackVertical(crop(0,0,0,2).Tweak(sat=2,dither=true),crop(0,2,0,0))
f3=StackVertical(crop(0,0,0,1036),crop(0,1036,0,0).FixRowBrightness(0,16,235,16,16))
f4=StackVertical(crop(0,0,0,1032),crop(0,1032,0,0).FixRowBrightness(1,16,235,16,16).FixRowBrightness(2,16,235,16,16).FixRowBrightness(3,16,235,16,16))
ConditionalFilter(f1, last, "SceneType", "=", "1")
ConditionalFilter(f2, last, "SceneType", "=", "2")
ConditionalFilter(f3, last, "SceneType", "=", "3")
ConditionalFilter(f4, last, "SceneType", "=", "4")
ConditionalReader("fix.txt", "SceneType")

fix.txt

type int
default 0

R 333 660 1
R 2225 2245 1
R 6727 6752 1
R 6926 7082 1
R 7189 7363 1
R 7501 7506 1
R 7734 7864 1
R 8037 8074 1
R 19350 19446 1
R 24017 24110 1
R 34155 34725 1
R 35277 35364 1
R 35430 35554 1
R 42816 42857 1
R 45216 45358 1
R 55019 55042 1
R 55207 55401 2
R 56530 56642 3
R 57855 57963 1
R 58772 58980 1
R 60380 60447 4
R 67604 67653 1
R 80595 80715 1
R 90608 90633 1
R 102761 102826 1
R 102909 102976 1
R 103018 103059 1
R 104931 104941 1
R 108812 108891 1
R 115942 115967 1
R 116027 116160 1
R 116718 116748 1
R 116790 116887 1
R 116967 117057 1
R 131975 131989 1
