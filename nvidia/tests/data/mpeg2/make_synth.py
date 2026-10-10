#!/usr/bin/env python3
"""Writes the synthetic MPEG-2 fixtures (see m2gen.py): streams FFmpeg's encoder cannot make -- field pictures, 16x8 and dual prime motion
compensation, concealment motion vectors, pulldown flags, several slices per row, odd stream structures and header variations. The
pictures are random syntax; what the card's and VirtualGPU's decoders make of them is compared by nvcuvid_mpeg2.cpp and
test_mpeg2_decode.cpp. Deterministic: the same files come out every time.

usage: make_synth.py    (writes *.m2v next to this script)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2gen   # noqa: E402

W, H = 96, 64
out = {}


def stream(prog=True, w=W, h=H, seed=7, **kw):
    s = m2gen.Stream(w, h, progressive_sequence=prog, seed=seed, **kw)
    s.sequence_header()
    return s


def field_pair(s, ptype, tr, tff=True, second_type=None, **extra):
    """A frame coded as two field pictures; the first field is the top one when tff."""
    first = 1 if tff else 2
    second = 3 - first
    st = second_type or ptype
    s.picture(ptype, tr, structure=first, tff=tff, prog_frame=0, **extra)
    ex = dict(extra)
    if ptype == 1 and st == 2:
        # the second field of an I frame predicts from the first (7.6.3.5): only the opposite parity, no skipped macroblocks, no dual prime
        ex.update(second_after_intra=True, restrict_intra_ref=True, no_skip=True, no_dual=True)
    s.picture(st, tr, structure=second, tff=tff, prog_frame=0, **ex)


def frame(s, ptype, tr, tff=True, **extra):
    s.picture(ptype, tr, structure=3, tff=tff, prog_frame=0, **extra)


# ---- field pictures and interlaced tools ---------------------------------------------------------------------------------------
def fields(tff, seed):
    s = stream(prog=False, seed=seed)
    s.gop()
    field_pair(s, 1, 0, tff, second_type=2)
    field_pair(s, 2, 3, tff)
    field_pair(s, 3, 1, tff)
    field_pair(s, 3, 2, tff)
    field_pair(s, 2, 6, tff)
    field_pair(s, 3, 4, tff)
    field_pair(s, 3, 5, tff)
    return s.bytes()


out['fields_ipb_tff'] = fields(True, 11)
out['fields_ipb_bff'] = fields(False, 12)


def fields_ip():
    s = stream(prog=False, seed=13)
    s.gop()
    field_pair(s, 1, 0, True, second_type=2)
    for i in range(1, 5):
        field_pair(s, 2, i, True)
    return s.bytes()


out['fields_ip'] = fields_ip()


def mixed():
    s = stream(prog=False, seed=14)
    s.gop()
    frame(s, 1, 0)
    field_pair(s, 2, 3)
    frame(s, 3, 1)
    field_pair(s, 3, 2)
    frame(s, 2, 6)
    field_pair(s, 3, 4)
    frame(s, 3, 5)
    field_pair(s, 2, 7, tff=False)
    return s.bytes()


out['fields_mixed'] = mixed()


def interlaced_frames(seed, name, **kw):
    s = stream(prog=False, seed=seed)
    s.gop()
    frame(s, 1, 0, **kw)
    frame(s, 2, 3, **kw)
    frame(s, 3, 1, **kw)
    frame(s, 3, 2, **kw)
    frame(s, 2, 6, **kw)
    frame(s, 3, 4, **kw)
    frame(s, 3, 5, **kw)
    out[name] = s.bytes()


interlaced_frames(21, 'frames_interlaced_tools')
interlaced_frames(22, 'frames_interlaced_bff', tff=False)
interlaced_frames(23, 'frames_slices', slices_per_row=3)
interlaced_frames(24, 'frames_junk', junk=True)
interlaced_frames(25, 'frames_escapes', level_cap=100, slice_qcode=2)


def dual_prime():
    s = stream(prog=False, seed=26)
    s.gop()
    frame(s, 1, 0)
    for i in range(1, 7):
        frame(s, 2, i, tff=bool(i % 2))
    return s.bytes()


out['dual_prime_frames'] = dual_prime()


def dual_prime_fields():
    s = stream(prog=False, seed=27)
    s.gop()
    field_pair(s, 1, 0, True, second_type=2)
    for i in range(1, 5):
        field_pair(s, 2, i, bool(i % 2))
    return s.bytes()


out['dual_prime_fields'] = dual_prime_fields()


def progressive_tools():
    s = stream(prog=True, seed=28)
    s.gop()
    s.picture(1, 0)
    s.picture(2, 3)
    s.picture(3, 1)
    s.picture(3, 2)
    s.picture(2, 6)
    s.picture(3, 4)
    s.picture(3, 5)
    return s.bytes()


out['progressive_tools'] = progressive_tools()


def conceal():
    s = stream(prog=False, seed=29)
    s.gop()
    frame(s, 1, 0, concealment=True)
    frame(s, 2, 3, concealment=True)
    frame(s, 3, 1, concealment=True)
    field_pair(s, 2, 6, concealment=True)
    field_pair(s, 3, 4, concealment=True)
    return s.bytes()


out['conceal'] = conceal()


def coding_ext():
    """every combination of the picture coding extension tools that changes how the data is read"""
    s = stream(prog=False, seed=30)
    s.gop()
    frame(s, 1, 0, intra_dc_precision=3, intra_vlc_format=1, alternate_scan=1, q_scale_type=1)
    frame(s, 2, 3, intra_dc_precision=2, alternate_scan=1)
    frame(s, 3, 1, intra_dc_precision=1, q_scale_type=1)
    frame(s, 3, 2, intra_dc_precision=0, intra_vlc_format=1)
    frame(s, 2, 6, intra_dc_precision=3, q_scale_type=1, intra_vlc_format=1)
    frame(s, 1, 7, intra_dc_precision=1, alternate_scan=1)
    field_pair(s, 2, 8, intra_dc_precision=2, alternate_scan=1, q_scale_type=1)
    return s.bytes()


out['coding_ext'] = coding_ext()


def quant_ext():
    s = stream(prog=True, seed=31)
    s.gop()
    m1 = [8] + [((i * 7) % 23) + 9 for i in range(1, 64)]
    m2 = [((i * 5) % 19) + 10 for i in range(64)]
    s.picture(1, 0, quant_ext=dict(intra=m1, inter=m2))
    s.picture(2, 1)
    s.picture(2, 2, quant_ext=dict(inter=[20] * 64))
    s.picture(1, 3, quant_ext=dict(intra=[8] + [16] * 63))
    return s.bytes()


out['quant_ext'] = quant_ext()


def long_vectors():
    s = stream(prog=True, w=128, h=96, seed=32)
    s.gop()
    s.picture(1, 0)
    s.picture(2, 3, f_code=[[6, 6], [15, 15]])
    s.picture(3, 1, f_code=[[6, 6], [6, 6]])
    s.picture(3, 2, f_code=[[1, 1], [2, 2]])
    return s.bytes()


out['long_vectors'] = long_vectors()

# ---- repeat flags ------------------------------------------------------------------------------------------------------------------
def rff_prog():
    s = stream(prog=True, seed=41)
    s.gop()
    s.picture(1, 0, prog_frame=1, repeat_first_field=True, tff=False)
    s.picture(2, 1, prog_frame=1, repeat_first_field=True, tff=True)
    s.picture(2, 2, prog_frame=1, repeat_first_field=False, tff=True)
    s.picture(2, 3, prog_frame=1, repeat_first_field=True, tff=False)
    return s.bytes()


out['rff_prog'] = rff_prog()


def pulldown():
    s = stream(prog=False, seed=42)
    s.gop()
    for i in range(8):
        s.picture(1 if i == 0 else 2, i, prog_frame=1, repeat_first_field=(i % 2 == 0), tff=(i % 4 < 2), frame_pred_frame_dct=1)
    return s.bytes()


out['pulldown'] = pulldown()


def rff_interlaced():
    s = stream(prog=False, seed=43)
    s.gop()
    s.picture(1, 0, prog_frame=0, repeat_first_field=True, tff=True)
    s.picture(2, 1, prog_frame=0, repeat_first_field=True, tff=False)
    s.picture(2, 2, prog_frame=0, repeat_first_field=False, tff=False)
    return s.bytes()


out['rff_interlaced'] = rff_interlaced()

# ---- stream structure ------------------------------------------------------------------------------------------------------------------
def unpaired():
    s = stream(prog=False, seed=51)
    s.gop()
    s.picture(1, 0, structure=1, tff=True, prog_frame=0)
    s.picture(2, 1, structure=3, tff=True, prog_frame=0)
    s.picture(2, 2, structure=3, tff=True, prog_frame=0)
    return s.bytes()


out['unpaired_field'] = unpaired()


def twotop():
    s = stream(prog=False, seed=52)
    s.gop()
    s.picture(1, 0, structure=1, tff=True, prog_frame=0)
    s.picture(1, 1, structure=1, tff=True, prog_frame=0)
    s.picture(2, 2, structure=3, tff=True, prog_frame=0)
    return s.bytes()


out['two_top_fields'] = twotop()


def bff_fields():
    s = stream(prog=False, seed=53)
    s.gop()
    field_pair(s, 1, 0, False, second_type=2)
    field_pair(s, 2, 1, False)
    return s.bytes()


out['bottom_first_fields'] = bff_fields()


def lead_b():
    s = stream(prog=True, seed=54)
    s.gop(closed=False, broken=True)
    s.picture(1, 2)
    s.picture(3, 0)
    s.picture(3, 1)
    s.picture(2, 5)
    s.picture(3, 3)
    s.picture(3, 4)
    return s.bytes()


out['leading_b'] = lead_b()


def start_p():
    s = stream(prog=True, seed=55)
    s.gop()
    s.picture(2, 0)
    s.picture(2, 1)
    s.picture(1, 2)
    s.picture(2, 3)
    return s.bytes()


out['start_with_p'] = start_p()


def temporal_reference():
    s = stream(prog=True, seed=56)
    s.gop()
    s.picture(1, 7)
    s.picture(2, 3)
    s.picture(3, 100)
    s.picture(3, 5)
    s.picture(2, 1)
    return s.bytes()


out['temporal_reference'] = temporal_reference()


def seq(w=W, h=H, prog=True, n=3, seed=60, **kw):
    s = m2gen.Stream(w, h, progressive_sequence=prog, seed=seed, **kw)
    s.sequence_header()
    s.gop()
    s.picture(1, 0, prog_frame=1 if prog else 0)
    for i in range(1, n):
        s.picture(2, i, prog_frame=1 if prog else 0)
    return s


out['seq_size_change'] = seq().bytes() + seq(w=64, h=48, seed=61).bytes()
out['seq_height_change'] = seq().bytes() + seq(h=48).bytes()
out['seq_end_same'] = seq().bytes() + bytes([0, 0, 1, 0xB7]) + seq().bytes()
out['seq_repeated'] = seq().bytes() + seq().bytes()
out['seq_aspect_change'] = seq().bytes() + seq(aspect=3).bytes()
out['seq_rate_change'] = seq().bytes() + seq(frame_rate_code=3).bytes()
out['seq_bitrate_change'] = seq().bytes() + seq(bit_rate=5000).bytes()
out['seq_progressive_change'] = seq().bytes() + seq(prog=False).bytes()
out['seq_display_change'] = seq().bytes() + seq(display=dict(w=W, h=H, video_format=2)).bytes()
out['seq_lowdelay_change'] = seq().bytes() + seq(low_delay=1).bytes()
out['seq_matrix_change'] = seq().bytes() + seq(intra_matrix=[8] + [20] * 63).bytes()
out['seq_chroma422_then_good'] = seq(chroma_format=2).bytes() + seq().bytes()
out['seq_chroma422_twice'] = seq(chroma_format=2).bytes() + seq(chroma_format=2).bytes()
out['seq_rate0_then_good'] = seq(frame_rate_code=0).bytes() + seq().bytes()

# ---- header fields ---------------------------------------------------------------------------------------------------------------------
for a in (1, 2, 3, 4, 5):
    out['hdr_aspect_%d' % a] = seq(n=2, aspect=a).bytes()
for r in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
    out['hdr_rate_%d' % r] = seq(n=2, frame_rate_code=r).bytes()
out['hdr_rate_ext'] = seq(n=2, frame_rate_code=4, rate_ext=(1, 3)).bytes()
for cf in (0, 2, 3):
    out['hdr_chroma_%d' % cf] = seq(n=2, chroma_format=cf).bytes()
out['hdr_display_colour'] = seq(n=2, display=dict(video_format=3, colour=(5, 6, 7), w=W, h=H)).bytes()
out['hdr_display_nocolour'] = seq(n=2, display=dict(video_format=2, w=W, h=H)).bytes()
out['hdr_display_size'] = seq(n=2, display=dict(video_format=5, w=80, h=48)).bytes()
out['hdr_bitrate'] = seq(n=2, bit_rate=0x12345).bytes()
out['hdr_no_extension'] = seq(n=2, no_seq_ext=True).bytes()
out['hdr_low_delay'] = seq(n=2, low_delay=1).bytes()

# sizes (the width is a whole number of macroblocks: the card decodes the rest of a wider picture to garbage; the height is any)
for (w, h, prog) in ((48, 32, True), (64, 17, True), (64, 17, False), (80, 33, True), (80, 33, False), (112, 112, True), (64, 100, False)):
    out['size_%dx%d_%s' % (w, h, 'p' if prog else 'i')] = seq(w=w, h=h, prog=prog, n=2, seed=70 + w + h).bytes()

for name, data in sorted(out.items()):
    with open(os.path.join(HERE, name + '.m2v'), 'wb') as f:
        f.write(data)
print('%d synthetic fixtures' % len(out))

if '--stats' in sys.argv:
    for k, v in sorted(m2gen.STATS.items()):
        print(v, k)
