#!/usr/bin/env bash
# NVENC through the bare soname applications dlopen: libvgpunvenc in the shim
# directory, standing in for the driver's libnvidia-encode.so.1.
#
#   nvenc_encode.cu  an encode through the API, the stream checked for shape
#   nvenc_api.cpp    every status, count, capability, preset and error string of
#                    the API, compared with what an RTX 3060 printed
#                    (nvenc_api.rtx3060.txt)
#   nvenc_h264.cpp   frames encoded in each input format and decoded back with
#                    ffmpeg, which must return the input (lossless I_PCM stream)
#
# --card   run nvenc_api and nvenc_h264 against the driver's own library on a
#          real card instead (LD_LIBRARY_PATH from VGPU_CARD_LIBS, default
#          /usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib): nvenc_api must still
#          print the golden file, nvenc_h264 runs with --lossy
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
trap 'rm -f "$out" "$out"_api "$out"_h264 "$out".txt' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ $mode == sim && ! -e "$shim/libnvidia-encode.so.1" ]]; then
  echo "SKIP: libvgpunvenc not built (nvEncodeAPI.h absent at build time)"; exit 0
fi

flags=(-w -std=c++17 -cudart shared -arch=compute_86 -code=compute_86 -Wno-deprecated-gpu-targets -I "$inc")
if [[ $mode == card ]]; then
  env_prefix=(env "LD_LIBRARY_PATH=${VGPU_CARD_LIBS:-/usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib}")
  lossy=(--lossy)
else
  # shellcheck disable=SC2207
  flags+=($(shim_sanitizer_nvcc_flags "$shim") -L "$shim")
  # libcuda and libcudart both load (cudart shared): both_shims_env is what a sanitizer build needs for that.
  # shellcheck disable=SC2046
  env_prefix=(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 "LD_LIBRARY_PATH=$shim")
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
if ! require_shim_libs "$shim" "${out}_api"; then exit 0; fi
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
if ! require_shim_libs "$shim" "${out}_h264"; then exit 0; fi
h264="$("${env_prefix[@]}" "${out}_h264" "${lossy[@]}" 2>&1)" || { echo "$h264"; fail=1; }
echo "$h264"
[[ "$h264" == SKIP* || "$h264" == *PASS* ]] || fail=1
[[ $fail == 0 ]] && echo "PASS"
exit $fail
