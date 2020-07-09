import vapoursynth as vs
core = vs.core
import vsutil as vsu
import awsmfunc as awf
import fvsfunc as fvf
import kagefunc as kgf
from rekt import *
from adptvgrnMod import *
from functools import partial


def rektlvls(clip, rownum=None, rowval=None, colnum=None, colval=None, prot_val=[16, 235], min=16, max=235, autoprot=False, autotol=[1, 2]):
    '''
    Wrapper around rektlvl: a rekt_fast version of havsfunc's FixBrightnessProtect2.
    :param clip: Clip to be processed.
    :param rownum: Row(s) to be processed.
    :param rowval: Row adjustment value. Negatives darken, positives brighten. Values can be between -100 and 100.
    :param colnum: Column(s) to be processed.
    :param colval: Column adjustment value. Negatives darken, positives brighten. Values can be between -100 and 100.
    :param prot_val: If None, this will work like FixBrightness. If an int, values above 255 - prot_val will not be
                     processed. If list, first int is value below which no processing takes place, second int is same as
                     no list.
    :return: Clip with first plane's values adjusted by adj_val.
    '''
    if autoprot:
        def findprot(n, f, clip=clip, rownum=rownum, rowval=rowval, colnum=colnum, colval=colval):
            prot_val = [f.props.PlaneStatsMin >> (clip.format.bits_per_sample - 8) + autotol[0], f.props.PlaneStatsMax >> (clip.format.bits_per_sample - 8) - autotol[1]]
            return rektlvls(clip, rownum=rownum, rowval=rowval, colnum=colnum, colval=colval, prot_val=prot_val)
        return clip.std.FrameEval(partial(findprot), clip.std.PlaneStats())
    else:
        if rownum is not None:
            if isinstance(rownum, int):
                rownum = [rownum]
            if isinstance(rowval, int):
                rowval = [rowval]
            for _ in range(len(rownum)):
                clip = rektlvl(clip, rownum[_], rowval[_], type='row', prot_val=prot_val, min=min, max=max)
        if colnum is not None:
            if isinstance(colnum, int):
                colnum = [colnum]
            if isinstance(colval, int):
                colval = [colval]
            for _ in range(len(colnum)):
                clip = rektlvl(clip, colnum[_], colval[_], type='column', prot_val=prot_val, min=min, max=max)
        return clip


src = core.ffms2.Source("The.World.Of.Kanako.2014.1080p.BluRay.REMUX.AVC.DTS-HD.MA5.1-MC.mkv").std.Crop(top=138, bottom=138)

src = vsu.depth(src, 16)

##################################################################
# The required prot_val varies by frame, so I just automated it. #
##################################################################

src = rektlvls(src, rownum=[0, 1, 2, 3, 4, 5, 797, 800, 801, 802], rowval=[13, -8, 6, -1, -1, 2, 2, 5, -9, 13], autoprot=True, autotol=[1, 2])

############################################################
# Dirty chroma at the top and bottom requires fillborders. #
############################################################

src = awf.fb(src, bottom=1, planes=[1, 2], top=1)
src = fixbrdrs(src, bottom=True)

##############################################################################################
# I would rather encode this in 1080p, but the required bitrate is above that of the source. #
##############################################################################################

src = awf.cr(src, 720, bottom=1)

###############################################
# Fix for intro banding, gf3 just works here. #
###############################################

dbn = fvf.gf3(src, thr=.5, elast=4, radius=25)
dbn = adptvgrnMod(dbn, luma_scaling=10, size=1, strength=.1, seed=222, static=True, lo=19)
src = awf.rfs(src, dbn, "[24 407]")

######################################################################################################
# The source has tons of issues with banding, blocking, and lost grain.                              #
# The easiest way to solve this was to use one f3kdb along a sizedgrn call with the following masks: #
#  - One edge mask to keep details from being destroyed by f3kdb.                                    #
#  - One bandmask call to keep f3kdb from affecting issue-free areas.                                # 
#  - One luma-adaptive mask on bandmask to only apply grain to very flat areas.                      #
#  - One luma-adaptive mask to limit f3kdb and sizedgrn to dark areas of dark frames.                #
######################################################################################################

dbn = src.f3kdb.Deband(range=10, y=48, cb=32, cr=32, grainy=0, grainc=0, output_depth=16)
msk = kgf.retinex_edgemask(dbn).std.Expr("x 5000 > x 4 * x ?").std.Maximum().std.Inflate().std.Maximum()
msk.set_output()
dbn = dbn.std.MaskedMerge(src, dbn)
msk = awf.bandmask(src, thr=200)
dbn = src.std.MaskedMerge(dbn, msk)
grn = sizedgrn(dbn, size=1.25, strength=.6, cstrength=.4, sharp=80, seed=222, static=False)
msk = msk.std.Expr("x 0 = 49152 16384 ?")
msk = vsu.iterate(msk, core.std.Minimum, 6).std.PlaneStats().adg.Mask(luma_scaling=8)
dbn = dbn.std.MaskedMerge(grn, msk)
msk = src.std.PlaneStats().adg.Mask(luma_scaling=25)
dbn = src.std.MaskedMerge(dbn, msk)
src = awf.rfs(src, dbn, "[633 163056]")

###################################################################
# Remove dirty black frames at start and replace with clean ones. #
###################################################################

src = vsu.depth(src, 8).std.Trim(23)
src = src.std.BlankClip(color=[16, 128, 128], length=23) + src

src.set_output()
