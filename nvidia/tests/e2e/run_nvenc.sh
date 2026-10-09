#!/usr/bin/env bash
# NVENC through the bare soname applications dlopen: libvgpunvenc in the shim
# directory, standing in for the driver's libnvidia-encode.so.1.
#
#   nvenc_encode.cu  an encode through the API, the stream checked for shape
#   nvenc_api.cpp    every status, count, capability, preset and error string of
#                    the API, compared with what an RTX 3060 printed
#                    (nvenc_api.rtx3060.txt)
#   nvenc_h264.cpp   frames encoded in each input format and decoded back with
#                    ffmpeg: the compressed H.264 stream within a signal-to-noise
#                    floor, the lossless tuning and HEVC (PCM streams) exactly
#   nvenc_nvdec.cpp  H.264 streams (GOP structures, forced pictures, rate control,
#                    profiles, odd sizes) decoded by NVDEC (libnvcuvid) and by ffmpeg,
#                    which must agree to the sample
#
# --card   run nvenc_api, nvenc_h264 and nvenc_nvdec against the driver's own
#          libraries on a real card instead (LD_LIBRARY_PATH from VGPU_CARD_LIBS,
#          default /usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib): nvenc_api must
#          still print the golden file, the others run with --card
# --update with --card, rewrite the golden file from the card's output
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
here="$root/nvidia/tests/e2e"
inc="${NVENC_INCLUDE:-$root/nvidia/third_party/nvenc_include}"
golden="$here/nvenc_api.rtx3060.txt"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvenc_$$"
mode=sim update=0
for a in "$@"; do
  case "$a" in
    --card) mode=card ;;
    --update) update=1 ;;
    *) echo "unknown argument $a" >&2; exit 2 ;;
  esac
done
trap 'rm -f "$out" "$out"_api "$out"_h264 "$out"_nvdec "$out".txt' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ $mode == sim && ! -e "$shim/libnvidia-encode.so.1" ]]; then
  echo "SKIP: libvgpunvenc not built (nvEncodeAPI.h absent at build time)"; exit 0
fi

flags=(-w -std=c++17 -cudart shared -arch=compute_86 -code=compute_86 -Wno-deprecated-gpu-targets -I "$inc")
if [[ $mode == card ]]; then
  env_prefix=(env "LD_LIBRARY_PATH=${VGPU_CARD_LIBS:-/usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib}")
  lossy=(--card)
else
  # shellcheck disable=SC2207
  flags+=($(shim_sanitizer_nvcc_flags "$shim") -L "$shim")
  env_prefix=(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 "LD_LIBRARY_PATH=$shim")
  lossy=()
fi

fail=0
if [[ $mode == sim ]]; then
  # An RTX 3060: a card that has an encoder, unlike the datacenter A100.
  nvcc "${flags[@]}" "$here/nvenc_encode.cu" -o "$out" -ldl -lcuda
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  result="$("${env_prefix[@]}" "$out" 2>&1)" || { echo "$result"; exit 1; }
  echo "$result"
  [[ "$result" == *PASS* ]] || fail=1
fi

nvcc "${flags[@]}" "$here/nvenc_api.cpp" -o "${out}_api" -ldl -lcuda
if [[ $mode == sim ]] && ! require_shim_libs "$shim" "${out}_api"; then exit 0; fi
if "${env_prefix[@]}" "${out}_api" >"$out.txt" 2>/dev/null; then :; else echo "FAIL: nvenc_api exited $?"; fail=1; fi
if grep -q '^SKIP' "$out.txt"; then cat "$out.txt"; exit 0; fi
if [[ $mode == card && $update == 1 ]]; then
  cp "$out.txt" "$golden"; echo "wrote $golden"
elif diff -u "$golden" "$out.txt"; then
  echo "PASS nvenc_api: $(wc -l <"$out.txt") lines identical to the RTX 3060's"
else
  echo "FAIL nvenc_api: output differs from nvenc_api.rtx3060.txt"; fail=1
fi

nvcc "${flags[@]}" "$here/nvenc_h264.cpp" -o "${out}_h264" -ldl -lcuda
if [[ $mode == sim ]] && ! require_shim_libs "$shim" "${out}_h264"; then exit 0; fi
h264="$("${env_prefix[@]}" "${out}_h264" "${lossy[@]}" 2>&1)" || { echo "$h264"; fail=1; }
echo "$h264"
[[ "$h264" == SKIP* || "$h264" == *PASS* ]] || fail=1

# The same encoder's H.264 streams through NVDEC. In the simulator that needs the NVDEC shim too.
if [[ $mode == card || -e "$shim/libnvcuvid.so.1" ]]; then
  nvdec_inc="$root/nvidia/third_party/nvdec_include"
  nvcc "${flags[@]}" -I "$nvdec_inc" "$here/nvenc_nvdec.cpp" -o "${out}_nvdec" -ldl -lcuda
  if [[ $mode == sim ]] && ! require_shim_libs "$shim" "${out}_nvdec"; then exit 0; fi
  nvdec="$("${env_prefix[@]}" "${out}_nvdec" "${lossy[@]}" 2>&1)" || { echo "$nvdec"; fail=1; }
  echo "$nvdec"
  [[ "$nvdec" == SKIP* || "$nvdec" == *PASS* ]] || fail=1
else
  echo "SKIP nvenc_nvdec: libvgpunvcuvid not built"
fi
[[ $fail == 0 ]] && echo "PASS"
exit $fail
