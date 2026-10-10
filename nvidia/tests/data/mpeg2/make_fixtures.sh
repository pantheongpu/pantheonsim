#!/usr/bin/env bash
# Writes the MPEG-2 video fixtures nvcuvid_mpeg2.cpp and test_mpeg2_decode decode: short elementary streams from FFmpeg 6.1's
# mpeg2video encoder covering the coding tools of Main profile 4:2:0 frame pictures. The source is FFmpeg's testsrc2 (moving shapes,
# text, gradients) with a little noise added. The streams FFmpeg's encoder cannot make (field pictures, dual prime, 16x8 motion
# compensation, concealment motion vectors, pulldown flags, odd stream structures, header variations) come from make_synth.py.
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
gen i_only        96x64 3  "-g 1 -qscale:v 4"
gen p_only        96x64 8  "-bf 0 -g 12 -qscale:v 5"
gen b_frames      96x64 10 "-bf 2 -g 6 -qscale:v 5"
gen b_deep        96x64 12 "-bf 4 -g 15 -b_strategy 0 -qscale:v 5"
gen closed_gop    96x64 10 "-bf 2 -g 5 -flags +cgop -sc_threshold 1000000000 -qscale:v 5"
gen interlaced_tff 96x64 8 "-bf 2 -g 8 -flags +ildct+ilme -top 1 -qscale:v 5"
gen interlaced_bff 96x64 8 "-bf 2 -g 8 -flags +ildct+ilme -top 0 -qscale:v 5"
gen alt_scan      96x64 8  "-bf 2 -g 8 -flags +ildct+ilme -top 1 -alternate_scan 1 -qscale:v 5"
gen intra_vlc     96x64 6  "-bf 2 -g 8 -intra_vlc 1 -qscale:v 3"
gen nonlinear_q   96x64 6  "-bf 2 -g 8 -non_linear_quant 1 -qmax 28 -qscale:v 7"
gen dc9           96x64 5  "-bf 1 -g 8 -dc 9 -qscale:v 5"
gen dc10          96x64 5  "-bf 1 -g 8 -dc 10 -qscale:v 5"
gen dc11          96x64 5  "-bf 1 -g 8 -dc 11 -qscale:v 5"
gen matrices      96x64 6  "-bf 2 -g 8 -intra_matrix 8,16,19,22,26,27,29,34,16,16,22,24,27,29,34,37,19,22,26,27,29,34,34,38,22,22,26,27,29,34,37,40,22,26,27,29,32,35,40,48,26,27,29,32,35,40,48,58,26,27,29,34,38,46,56,69,27,29,35,38,46,56,69,83 -inter_matrix 16,17,18,19,20,21,22,23,17,18,19,20,21,22,23,24,18,19,20,21,22,23,24,25,19,20,21,22,23,24,25,26,20,21,22,23,24,25,26,27,21,22,23,24,25,26,27,28,22,23,24,25,26,27,28,29,23,24,25,26,27,28,29,30 -qscale:v 5"
gen lowq          96x64 6  "-bf 2 -g 8 -qscale:v 1"
gen highq         96x64 6  "-bf 2 -g 8 -qscale:v 28"
gen size_96x70    96x70 6  "-bf 2 -g 8 -qscale:v 5"
gen size_48x16    48x16 5  "-bf 1 -g 8 -qscale:v 5"
gen motion_fast   96x64 8  "-bf 2 -g 8 -qscale:v 5 -me_method full -bidir_refine 1" "tblend=all_mode=average,fps=30"
gen aspect_wide   96x64 4  "-bf 1 -g 8 -qscale:v 5 -aspect 16:9"
gen rate_25       96x64 4  "-bf 1 -g 8 -qscale:v 5 -r 25"
gen rate_2997     96x64 4  "-bf 1 -g 8 -qscale:v 5 -r 30000/1001"
gen vbr_bitrate   96x64 8  "-bf 2 -g 8 -b:v 100k -maxrate 150k -bufsize 100k"
gen seq_disp      96x64 4  "-bf 1 -g 8 -qscale:v 5 -seq_disp_ext 1 -video_format 1 -color_primaries bt709 -color_trc bt709 -colorspace bt709"
gen low_delay     96x64 6  "-bf 0 -g 8 -qscale:v 5 -flags +low_delay"
gen mbd_rd        96x64 6  "-bf 2 -g 8 -qscale:v 5 -mbd rd -trellis 1"
gen no_skip_b     96x64 6  "-bf 2 -g 8 -qscale:v 5 -mpv_flags +mv0"
gen timecode_gop  96x64 8  "-bf 2 -g 4 -qscale:v 5 -gop_timecode 01:02:03:04"
gen noise_hi      96x64 6  "-bf 2 -g 8 -qscale:v 6" "noise=alls=60:allf=t"
python3 make_synth.py
