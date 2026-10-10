#!/usr/bin/env bash
# Writes the MPEG-2 video fixtures nvcuvid_mpeg2.cpp and test_mpeg2_decode decode: short elementary streams from FFmpeg 6.1's
# mpeg2video encoder covering the coding tools of Main profile 4:2:0 frame pictures. The source is FFmpeg's testsrc2 (moving shapes,
# text, gradients) with a little noise added. A few fixtures are then edited bit-for-bit by make_variants.py (display flags, pulldown).
# They are committed: this script says how they were made, it is not run by the tests (the noise filter is seeded from the clock, so
# a rerun gives different, equally good pictures).
set -euo pipefail
cd "$(dirname "$0")"
gen() {   # name size frames "encoder options" [extra vf]
  local name=$1 size=$2 frames=$3 opts=$4 vf=${5:-}
  # shellcheck disable=SC2086
  ffmpeg -nostdin -v error -y -f lavfi -i "testsrc2=size=$size:rate=30,noise=alls=${NOISE:-8}:allf=t${vf:+,$vf},format=yuv420p" -frames:v "$frames" \
    -c:v mpeg2video -threads 1 $opts -f mpeg2video "$name.m2v"
}
gen i_only        192x144 4  "-g 1 -qscale:v 4"
gen p_only        192x144 10 "-bf 0 -g 12 -qscale:v 5"
gen b_frames      192x144 16 "-bf 2 -g 12 -qscale:v 5"
gen b_deep        192x144 20 "-bf 4 -g 15 -b_strategy 0 -qscale:v 5"
gen closed_gop    192x144 16 "-bf 2 -g 6 -flags +cgop -sc_threshold 1000000000 -qscale:v 5"
gen interlaced_tff 192x144 12 "-bf 2 -g 12 -flags +ildct+ilme -top 1 -qscale:v 5"
gen interlaced_bff 192x144 12 "-bf 2 -g 12 -flags +ildct+ilme -top 0 -qscale:v 5"
gen alt_scan      192x144 8  "-bf 2 -g 12 -flags +ildct+ilme -top 1 -alternate_scan 1 -qscale:v 5"
gen intra_vlc     192x144 8  "-bf 2 -g 12 -intra_vlc 1 -qscale:v 3"
gen nonlinear_q   192x144 8  "-bf 2 -g 12 -non_linear_quant 1 -qmax 28 -qscale:v 7"
gen dc9           192x144 6  "-bf 1 -g 12 -dc 9 -qscale:v 5"
gen dc10          192x144 6  "-bf 1 -g 12 -dc 10 -qscale:v 5"
gen dc11          192x144 6  "-bf 1 -g 12 -dc 11 -qscale:v 5"
gen matrices      192x144 8  "-bf 2 -g 12 -intra_matrix 8,16,19,22,26,27,29,34,16,16,22,24,27,29,34,37,19,22,26,27,29,34,34,38,22,22,26,27,29,34,37,40,22,26,27,29,32,35,40,48,26,27,29,32,35,40,48,58,26,27,29,34,38,46,56,69,27,29,35,38,46,56,69,83 -inter_matrix 16,17,18,19,20,21,22,23,17,18,19,20,21,22,23,24,18,19,20,21,22,23,24,25,19,20,21,22,23,24,25,26,20,21,22,23,24,25,26,27,21,22,23,24,25,26,27,28,22,23,24,25,26,27,28,29,23,24,25,26,27,28,29,30 -qscale:v 5"
gen lowq          192x144 8  "-bf 2 -g 12 -qscale:v 1"
gen highq         192x144 8  "-bf 2 -g 12 -qscale:v 28"
gen size_200x140  200x140 8  "-bf 2 -g 12 -qscale:v 5"
gen size_48x16    48x16 6    "-bf 1 -g 12 -qscale:v 5"
gen motion_fast   192x144 10 "-bf 2 -g 12 -qscale:v 5 -me_method full -bidir_refine 1" "tblend=all_mode=average,fps=30"
gen aspect_wide   192x144 6  "-bf 1 -g 12 -qscale:v 5 -aspect 16:9"
gen rate_25       192x144 6  "-bf 1 -g 12 -qscale:v 5 -r 25"
gen rate_2997     192x144 6  "-bf 1 -g 12 -qscale:v 5 -r 30000/1001"
gen rate_odd      192x144 6  "-bf 1 -g 12 -qscale:v 5 -r 7/1"
gen vbr_bitrate   192x144 12 "-bf 2 -g 12 -b:v 400k -maxrate 600k -bufsize 400k"
gen seq_disp      192x144 6  "-bf 1 -g 12 -qscale:v 5 -seq_disp_ext 1 -video_format 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709"
gen low_delay     192x144 8  "-bf 0 -g 12 -qscale:v 5 -flags +low_delay"
gen mbd_rd        192x144 8  "-bf 2 -g 12 -qscale:v 5 -mbd rd -trellis 1"
gen no_skip_b     192x144 8  "-bf 2 -g 12 -qscale:v 5 -mpv_flags +mv0"
gen timecode_gop  192x144 12 "-bf 2 -g 4 -qscale:v 5 -gop_timecode 01:02:03:04"
gen noise_hi      192x144 8  "-bf 2 -g 12 -qscale:v 6" "noise=alls=60:allf=t"
