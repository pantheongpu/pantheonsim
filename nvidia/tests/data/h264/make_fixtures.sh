#!/usr/bin/env bash
# Writes the H.264 fixtures nvcuvid_h264.cpp, test_h264_decode and test_nvenc_h264_lossy decode:
# short Annex B streams from x264 (libx264 in FFmpeg 6.1) covering the tools of
# Baseline, Main and High profile with 4:2:0 8-bit progressive and MBAFF
# pictures. The source is FFmpeg's testsrc2 (moving shapes, text, gradients) with a
# little noise added; x264 writes the streams. They are committed: this script
# says how they were made, it is not run by the tests (the noise filter
# is seeded from the clock, so a rerun gives different, equally good pictures).
set -euo pipefail
cd "$(dirname "$0")"
gen() {   # name size frames profile x264-params [extra vf]
  local name=$1 size=$2 frames=$3 profile=$4 params=$5 vf=${6:-}
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=$size:rate=30,noise=alls=${NOISE:-10}:allf=t${vf:+,$vf},format=yuv420p" -frames:v "$frames" \
    -c:v libx264 -profile:v "$profile" -x264-params "$params:threads=1:sliced-threads=0:aud=0" -bsf:v h264_mp4toannexb -f h264 "$name.h264"
}
gen i_cavlc        64x48   4  baseline "keyint=1:cabac=0:bframes=0:ref=1"
gen p_cavlc        96x64   10 baseline "cabac=0:bframes=0:ref=3:keyint=30:slices=3"
gen p_cabac        96x64   10 main     "cabac=1:bframes=0:ref=3:keyint=30"
gen b_spatial      96x64   14 main     "cabac=1:bframes=3:b-pyramid=normal:ref=4:direct=spatial:weightb=1:keyint=30"
gen b_temporal     96x64   14 main     "cabac=1:bframes=2:ref=3:direct=temporal:weightb=0:keyint=30"
gen b_cavlc        96x64   12 main     "cabac=0:bframes=3:b-pyramid=strict:ref=3:direct=auto:keyint=30"
gen high_8x8       112x80  12 high     "cabac=1:8x8dct=1:bframes=2:ref=3:partitions=all:subme=7:keyint=30"
gen high_cqm       80x64   10 high     "cabac=1:8x8dct=1:bframes=2:cqm=jvt:keyint=30"
gen high_cavlc_8x8 80x64   10 high     "cabac=0:8x8dct=1:bframes=2:ref=2:keyint=30"
gen weightp        96x64   12 main     "cabac=1:bframes=0:ref=2:weightp=2:keyint=30" "fade=in:0:12"
gen odd_size       70x38   8  high     "cabac=1:8x8dct=1:bframes=2:keyint=30"
gen lowqp          64x48   6  high     "cabac=1:8x8dct=1:bframes=1:qp=2:keyint=30"
gen lowqp_cavlc    64x48   6  main     "cabac=0:bframes=1:qp=2:keyint=30"
gen highqp         96x64   8  main     "cabac=1:bframes=2:crf=45:keyint=30"
gen mbaff          96x64   10 high     "cabac=1:8x8dct=1:bframes=2:tff=1:keyint=30"
gen mbaff_cavlc    96x64   10 main     "cabac=0:bframes=2:tff=1:keyint=30"
gen multislice_b   128x96  10 high     "cabac=1:8x8dct=1:bframes=2:slices=4:keyint=30"
gen long_gop       64x48   40 high     "cabac=1:bframes=2:ref=5:keyint=40:b-adapt=2"
gen deblock_off    64x48   6  main     "cabac=1:bframes=1:deblock=-1,-1"
gen deblock_strong 64x48   6  main     "cabac=1:bframes=1:deblock=6,6"

# Interlaced content, MBAFF coding with many field macroblock pairs: two instants of the moving test pattern woven into one frame
# (tinterlace), so that the pairs the encoder codes as field pairs are the ones where the instants differ.
gen_il() {   # name size rate frames profile x264-params
  local name=$1 size=$2 rate=$3 frames=$4 profile=$5 params=$6
  ffmpeg -nostdin -v error -y -f lavfi -i "testsrc2=size=$size:rate=$rate,noise=alls=${NOISE:-4}:allf=t,tinterlace=mode=interleave_top,format=yuv420p" -frames:v "$frames" \
    -c:v libx264 -profile:v "$profile" -x264-params "$params:threads=1:sliced-threads=0:aud=0" -bsf:v h264_mp4toannexb -f h264 "$name.h264"
}
gen_il mbaff_field          128x96 8 6  high "cabac=1:8x8dct=1:bframes=2:tff=1:keyint=30"
gen_il mbaff_field_cavlc    128x96 8 6  main "cabac=0:bframes=2:tff=1:keyint=30"
gen_il mbaff_field_intra    128x96 8 3  high "cabac=1:8x8dct=1:keyint=1:tff=1"
gen_il mbaff_field_temporal 128x96 8 6  main "cabac=1:bframes=3:direct=temporal:weightb=0:tff=1:b-pyramid=normal:ref=4:keyint=30"
gen_il mbaff_field_spatial  128x96 8 6  main "cabac=1:bframes=3:direct=spatial:weightb=1:tff=1:b-pyramid=strict:ref=3:keyint=30"
gen_il mbaff_field_wp       128x96 8 6  main "cabac=1:bframes=0:weightp=2:tff=1:ref=2:keyint=30"
gen_il mbaff_field_slices   128x96 8 6  high "cabac=1:8x8dct=1:bframes=2:tff=1:slices=3:keyint=30"
gen_il mbaff_field_bff      128x96 8 6  high "cabac=1:8x8dct=1:bframes=2:bff=1:keyint=30"
gen_il mbaff_field_crop     100x60 8 6 high "cabac=1:8x8dct=1:bframes=2:tff=1:keyint=30"

# A stream with several IDR pictures and B pictures around them.
gen idr_mid 64x48 20 main "cabac=1:bframes=2:ref=3:keyint=7:min-keyint=7:scenecut=0"
# The same stream with its sequence parameter set rewritten (sps_edit.py): no VUI at all, a VUI without timing
# information, a VUI without the bitstream restriction fields, and a higher level.
for v in "novui vui=none" "notiming vui=notiming" "norestr vui=norestr" "level40 level=40" "mdfb1 mdfb=1" "reorder1 reorder=1"; do
  set -- $v
  python3 sps_edit.py b_spatial.h264 sps_$1.h264 "${@:2}"
done
python3 sps_edit.py p_cavlc.h264 sps_p_novui.h264 vui=none
# streams for the display-delay measurements: no reordering with 1, 2, 4 and 6 reference pictures; two B-picture streams
for r in 1 2 4 6; do gen p_ref$r 64x48 14 main "cabac=1:bframes=0:ref=$r:keyint=100"; done
for r in 2 3; do gen b_ref$r 64x48 14 main "cabac=1:bframes=2:b-pyramid=none:ref=$r:keyint=100"; done

# Streams with extra NAL units, written with a few lines of Python from p_cabac / b_spatial (see make_nal_variants.py): an SEI
# message before every slice, an access unit delimiter before every slice, and the parameter sets (changed and unchanged) repeated
# before one of the last slices. The card's parser must give the same callbacks as for the plain streams.
python3 make_nal_variants.py
