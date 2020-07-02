import vapoursynth as vs
core = vs.core
import vsutil as vsu
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from rekt import *
from adptvgrnMod import *
from functools import partial

src = core.ffms2.Source("The.World.Of.Kanako.2014.1080p.BluRay.REMUX.AVC.DTS-HD.MA5.1-MC.mkv").std.Crop(top=138, bottom=138)

src = vsu.depth(src, 16)

##################################################################
# The required prot_val varies by frame, so I just automated it. #
##################################################################

def evalf(n, f, clip):
    return rektlvls(clip, rownum=[0, 1, 2, 3, 4, 5, 797, 800, 801, 802], rowval=[13, -8, 6, -1, -1, 2, 2, 5, -9, 13], prot_val=[f.props.PlaneStatsMin >> 8 + 1, f.props.PlaneStatsMax >> 8 - 2])

src = src.std.PlaneStats()
src = src.std.FrameEval(partial(evalf, clip=src), src)

############################################################
# Dirty chroma at the top and bottom requires fillborders. #
############################################################

src = awf.fb(src, bottom=1, planes=[1, 2], top=1)
src = awf.fb(src, bottom=1, planes=0)

##############################################################################################
# I would rather encode this in 1080p, but the required bitrate is above that of the source. #
##############################################################################################

src = awf.cr(src, 720, bottom=1)

###############################################
# Fix for intro banding, gf3 just works here. #
###############################################

dbn = fvf.gf3(src, thr=.5, elast=4, radius=25)
dbn = adptvgrnMod(dbn, luma_scaling=10, size=1, strength=.2, seed=222, static=True)
src = awf.rfs(src, dbn, "[24 407]")

######################################################################################################
# The source has tons of issues with banding, blocking, and lost grain.                              #
# The easiest way to solve this was to use one f3kdb along a sizedgrn call with the following masks: #
#  - One edge mask to keep details from being destroyed by f3kdb.                                    #
#  - One bandmask call to keep f3kdb from affecting issue-free areas.                                #
#  - One luma-adaptive mask to limit f3kdb and sizedgrn to dark areas of dark frames.                #
######################################################################################################

dbn = src.f3kdb.Deband(range=10, y=48, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
msk = kgf.retinex_edgemask(dbn).std.Expr("x 5000 > x 4 * x ?").std.Maximum().std.Inflate().std.Maximum()
msk.set_output()
dbn = dbn.std.MaskedMerge(src, dbn)
msk = awf.bandmask(src, thr=200)
dbn = src.std.MaskedMerge(dbn, msk)
dbn = sizedgrn(dbn, size=1.3, strength=.8, cstrength=.5, sharp=90, seed=222, static=False)
msk = src.adg.Mask(luma_scaling=40)
dbn = src.std.MaskedMerge(dbn, msk)
src = awf.rfs(src, dbn, "[633 163056]")

###################################################################
# Remove dirty black frames at start and replace with clean ones. #
###################################################################

src = vsu.depth(src, 8).std.Trim(23)
src = src.std.BlankClip(color=[16, 128, 128], length=23) + src

src.set_output()
