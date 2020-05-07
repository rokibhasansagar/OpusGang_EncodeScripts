import vapoursynth as vs
core = vs.get_core()

def mt_expand_multi(src, mode='rectangle', planes=None, sw=1, sh=1):
    if not isinstance(src, vs.VideoNode):
        raise TypeError('mt_expand_multi: This is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        return mt_expand_multi(core.std.Maximum(src, planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    else:
    return src

def rfs(clipa, clipb, mappings=None, filename=None):

    if not isinstance(clipa, vs.VideoNode):
        raise TypeError('ReplaceFrames: "clipa" must be a clip!')
    if not isinstance(clipb, vs.VideoNode):
        raise TypeError('ReplaceFrames: "clipb" must be a clip!')
    if clipa.format.id != clipb.format.id:
        raise TypeError('ReplaceFrames: "clipa" and "clipb" must have the same format!')
    if filename is not None and not isinstance(filename, str):
        raise TypeError('ReplaceFrames: "filename" must be a string!')
    if mappings is not None and not isinstance(mappings, str):
        raise TypeError('ReplaceFrames: "mappings" must be a string!')
    if mappings is None:
        mappings = ''

    if filename:
        with open(filename, 'r') as mf:
            mappings += '\n{}'.format(mf.read())
    # Some people used this as separators and wondered why it wasn't working
    mappings = mappings.replace(',', ' ').replace(':', ' ')

    frames = re.findall('\d+(?!\d*\s*\d*\s*\d*\])', mappings)
    ranges = re.findall('\[\s*\d+\s+\d+\s*\]', mappings)
    maps = []
    for range_ in ranges:
        maps.append([int(x) for x in range_.strip('[ ]').split()])
    for frame in frames:
        maps.append([int(frame), int(frame)])

    for start, end in maps:
        if start > end:
            raise ValueError('ReplaceFrames: Start frame is bigger than end frame: [{} {}]'.format(start, end))
        if end >= clipa.num_frames or end >= clipb.num_frames:
            raise ValueError('ReplaceFrames: End frame too big, one of the clips has less frames: {}'.format(end)) 

    out = clipa
    for start, end in maps:
        temp = clipb[start:end+1] 
        if start != 0:
            temp = out[:start] + temp
        if end < out.num_frames - 1:
            temp = temp + out[end+1:]
        out = temp
    return out

def debandmask(clip, lo=6144, hi=12288, lothr=320, hithr=384, mrad=2):
    core = vs.core
   
    f = clip.format
    bits = f.bits_per_sample
    isINT = f.sample_type==vs.INTEGER
   
    peak = (1 << bits) - 1 if isINT else 1
    clip = clip.std.ShufflePlanes(0, vs.GRAY)
   
    ma = mt_expand_multi(clip, mode='ellipse', sw=mrad, sh=mrad)
    mi = mt_inpand_multi(clip, mode='ellipse', sw=mrad, sh=mrad)
   
    rmask = core.std.Expr([ma, mi], 'x y -')
   
    mexpr = 'x {lo} < y {lothr} >= {peak} 0 ? x {hi} > y {hithr} >= {peak} 0 ? y x {lo} - {r} / {tr} * {lothr} + >= {peak} 0 ? ? ?'.format(lo=lo, hi=hi, lothr=lothr, hithr=hithr, peak=peak, r=hi-lo, tr=hithr-lothr)
   
    return core.std.Expr([clip, rmask], mexpr)

src = core.ffms2.Source("/home/aicha/Game.of.Thrones.S01E04.Cripples.Bastards.and.Broken.Things.2160p.UHD.BluRay.REMUX.HDR.HEVC.Atmos-EPSiLON.mkv")

src = core.resize.Spline36(src, format=vs.YUV420P16)

y = core.descale.Debilinear(src.std.ShufflePlanes(0, vs.GRAY), 1920, 1080)
u = src.std.ShufflePlanes(1, vs.GRAY)
v = src.std.ShufflePlanes(2, vs.GRAY)
src = core.std.ShufflePlanes([y, u, v], [0, 0, 0], vs.YUV444P16)
                     
b16 = src

db = b16.f3kdb.Deband(range=15, y=60, cb=60, cr=60, grainy=10, grainc=10, output_depth=16)
mask = debandmask(b16)
merged = core.std.MaskedMerge(db, b16, mask)
sfdb = rfs(b16, merged, mappings="[135 191]")

db = b16.f3kdb.Deband(range=15, y=25, cb=0, cr=0, grainy=32, grainc=15, output_depth=16, dynamic_grain=True)
mask = mask.std.Inflate()
mask2 = core.std.ShufflePlanes(b16, 0, vs.GRAY).std.Binarize(5000).std.Invert()
merged = core.std.MaskedMerge(b16, db, mask2)
merged = core.std.MaskedMerge(merged, b16, mask)
sfdb = rfs(sfdb, merged, mappings="[4434 4607] [8397 8460] [8594 8646] [8696 8778] [8968 9060] [9419 9486] [30009 30222] [30275 30584] [37466 37534] [37585 37644] [76006 76287] [76396 76502] [76622 76886] [77062 77340] [77546 77752] [78002 78077] [78228 78339]")

final = core.resize.Spline36(sfdb, format=vs.YUV444P10)

final.set_output()