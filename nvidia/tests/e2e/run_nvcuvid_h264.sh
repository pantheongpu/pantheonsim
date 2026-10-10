#!/usr/bin/env bash
# NVDEC's H.264 decoder and video parser through the bare soname applications link.
#
# nvcuvid_h264.cpp prints the decoder capabilities, the decoder-creation answers, and,
# for every stream in nvidia/tests/data/h264, each callback of the video parser (sequence
# format, picture parameters, display order, timestamps) plus the CRC-32 of every displayed
# NV12 frame; nvcuvid_h264.rtx3060.txt is what the card printed. H.264 decoding is
# bit-exact by definition, so the checksums are the card's.
#
# --card    run against the driver's own library on a real card instead
#           (LD_LIBRARY_PATH from VGPU_CARD_LIBS, default
#           /usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib): the output must still be
#           the golden file
# --update  with --card, rewrite the golden file and the pixel files of the rescaled
#           streams from the card
# VGPU_H264_SKIP=a,b   streams to leave out (sections of the golden file and of the output are not compared); none by default
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
here="$root/nvidia/tests/e2e"
golden="$here/nvcuvid_h264.rtx3060.txt"
data="$root/nvidia/tests/data"
inc="$root/nvidia/third_party/nvdec_include"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvcuvid_h264_$$"
mode=sim update=0
skip="${VGPU_H264_SKIP:-}"
for a in "$@"; do
  case "$a" in
    --card) mode=card; skip="" ;;
    --update) update=1 ;;
    *) echo "unknown argument $a" >&2; exit 2 ;;
  esac
done
trap 'rm -rf "$out" "$out".txt "$out".err "$out".lib "$out".want' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
flags=(-w -std=c++17 -cudart none -arch=compute_86 -code=compute_86 -Wno-deprecated-gpu-targets -I "$inc")
args=()
if [[ $mode == card ]]; then
  libdir="${VGPU_CARD_LIBS:-/usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib}"
  mkdir -p "$out.lib"
  for d in ${libdir//:/ }; do
    [[ -e "$d/libnvcuvid.so.1" ]] && ln -sf "$d/libnvcuvid.so.1" "$out.lib/libnvcuvid.so" && break
  done
  [[ -e "$out.lib/libnvcuvid.so" ]] || { echo "SKIP: no libnvcuvid.so.1 in $libdir"; exit 0; }
  flags+=(-L "$out.lib" -L /usr/lib/wsl/lib)
  env_prefix=(env "LD_LIBRARY_PATH=$libdir" "VGPU_E2E_DATA=$data")
  if [[ $update == 1 ]]; then mkdir -p "$data/nvdec"; args=(--update "$data/nvdec"); fi
else
  if [[ ! -e "$shim/libnvcuvid.so.1" ]]; then
    echo "SKIP: libvgpunvcuvid not built (the CUDA ABI headers were absent at build time)"; exit 0
  fi
  # The shim holds libnvcuvid.so.1 only, like the driver; -lnvcuvid wants the unversioned name (a machine with the
  # driver's dev symlink found that one, a CI runner without it failed to link), so name it from a directory of ours.
  mkdir -p "$out.lib"
  ln -sf "$shim/libnvcuvid.so.1" "$out.lib/libnvcuvid.so"
  # shellcheck disable=SC2207
  flags+=($(shim_sanitizer_nvcc_flags "$shim") -L "$out.lib" -L "$shim")
  # libnvcuvid and libcuda each carry the simulator's core: both_shims_env is what a sanitizer build needs for that.
  # shellcheck disable=SC2046
  env_prefix=(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 "LD_LIBRARY_PATH=$shim" "VGPU_E2E_DATA=$data")
  [[ -n $skip ]] && args+=(--skip "$skip")
fi
nvcc "${flags[@]}" "$here/nvcuvid_h264.cpp" -o "$out" -ldl -lcuda -lnvcuvid
if [[ $mode == sim ]] && ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
"${env_prefix[@]}" "$out" --tolerance 1 "${args[@]}" "$data" >"$out.txt" 2>"$out.err" || status=$?
if [[ $mode == card && $update == 1 ]]; then
  cp "$out.txt" "$golden"; echo "wrote $golden and $data/nvdec"; exit 0
fi
# Leave the skipped streams' sections out of the golden file.
awk -v skip=",$skip," '
  /^stream / { name = $2; drop = index(skip, "," name ",") > 0 }
  /^done/ { drop = 0 }
  !drop { print }' "$golden" >"$out.want"
if diff -u "$out.want" "$out.txt"; then
  echo "PASS nvcuvid_h264: $(wc -l <"$out.txt") lines identical to the RTX 3060's"
else
  echo "FAIL nvcuvid_h264: output differs from nvcuvid_h264.rtx3060.txt"; cat "$out.err"; exit 1
fi
cat "$out.err"
[[ $status == 0 ]] && echo PASS
