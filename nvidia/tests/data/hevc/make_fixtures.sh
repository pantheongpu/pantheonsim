#!/usr/bin/env bash
# Writes the HEVC fixtures nvcuvid_hevc.cpp, test_hevc_decode and the NVENC tests decode: short Annex B streams from x265
# (libx265 in FFmpeg 6.1) covering the tools of Main and Main 10 profile 4:2:0 progressive pictures, and streams from the
# RTX 3060's NVENC (hevc_nvenc). The source is FFmpeg's testsrc2 (moving shapes, text, gradients) with a little noise added.
# They are committed: this script says how they were made, it is not run by the tests (the noise filter is seeded from the
# clock, so a rerun gives different, equally good pictures).
set -euo pipefail
cd "$(dirname "$0")"
gen() {   # name size frames x265-params [extra vf] [pix_fmt] [threading]
  local name=$1 size=$2 frames=$3 params=$4 vf=${5:-} pix=${6:-yuv420p} threading=${7:-pools=none:frame-threads=1}
  ffmpeg -nostdin -v error -y -f lavfi -i "testsrc2=size=$size:rate=30,noise=alls=${NOISE:-8}:allf=t${vf:+,$vf},format=$pix" -frames:v "$frames" \
    -c:v libx265 -x265-params "$params:log-level=error:$threading:repeat-headers=1" -f hevc "$name.h265"
}
gen i_only        192x144 3  "keyint=1"
gen p_low         192x144 10 "bframes=0:ref=3:keyint=30"
gen b_pyramid     192x144 16 "bframes=4:b-pyramid=1:ref=4:keyint=40"
gen b_flat        192x144 12 "bframes=3:b-pyramid=0:ref=3:keyint=30"
gen sao_off       160x144 8  "bframes=2:no-sao=1:keyint=30"
gen deblock_off   160x144 8  "bframes=2:deblock=0:keyint=30"
gen deblock_offs  160x144 8  "bframes=2:deblock=-3,3:keyint=30"
gen amp_rect      192x144 10 "bframes=2:amp=1:rect=1:keyint=30"
gen ctu16         192x144 8  "bframes=2:ctu=16:min-cu-size=8:keyint=30"
gen ctu32         192x144 8  "bframes=2:ctu=32:min-cu-size=8:keyint=30"
gen tskip         192x144 8  "bframes=2:tskip=1:tskip-fast=0:keyint=30"
gen lossless      160x144 4  "lossless=1:bframes=1:keyint=30"
gen cu_lossless   160x144 6  "cu-lossless=1:bframes=1:keyint=30:qp=30"
gen no_signhide   160x144 8  "bframes=2:signhide=0:keyint=30"
gen scaling_def   160x144 8  "bframes=2:scaling-list=default:keyint=30"
gen wpp           192x144 8  "bframes=2:wpp=1:keyint=30"
# x265 writes empty slices with its thread pool off: let it have the pool (one frame thread keeps the output deterministic)
gen slices        192x144 8  "bframes=2:slices=3:keyint=30" "" yuv420p "frame-threads=1"
gen weightp       160x144 12 "bframes=0:weightp=1:ref=2:keyint=30" "fade=in:0:12"
gen weightb       160x144 12 "bframes=2:weightb=1:weightp=1:ref=3:keyint=30" "fade=in:0:12"
gen no_tmvp       160x144 8  "bframes=2:temporal-mvp=0:keyint=30"
gen main10        192x144 10 "bframes=2:keyint=30:output-depth=10" "" yuv420p10le
gen main10_i      160x144 3  "keyint=1:output-depth=10" "" yuv420p10le
gen main10_wide   288x144 5  "bframes=2:keyint=30:output-depth=10" "" yuv420p10le
gen crop          198x138 8  "bframes=2:keyint=30"
gen crop_odd      202x146 6  "bframes=2:keyint=30"
gen size_136      160x136 4  "bframes=1:keyint=30"
gen cip           160x144 8  "bframes=2:constrained-intra=1:keyint=30"
gen no_strong     192x144 4  "keyint=1:strong-intra-smoothing=0"
gen open_gop      192x144 30 "bframes=3:open-gop=1:keyint=12:min-keyint=12:scenecut=0"
gen long_gop      176x144 40 "bframes=3:ref=5:keyint=40:b-adapt=2"
gen lowqp         160x144 5  "bframes=1:qp=2:keyint=30"
gen highqp        160x144 8  "bframes=2:crf=45:keyint=30"
gen aud_hrd       160x144 8  "bframes=2:aud=1:hrd=1:keyint=30"
gen max_merge2    160x144 8  "bframes=2:max-merge=2:keyint=30"
gen cutree_idr    160x144 20 "bframes=2:ref=3:keyint=7:min-keyint=7:scenecut=0"
gen rd_deep       192x144 6  "bframes=2:rd=6:rdoq-level=2:keyint=30"
gen tu_deep       192x144 6  "bframes=2:tu-intra-depth=4:tu-inter-depth=4:keyint=30"
