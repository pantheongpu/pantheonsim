#!/usr/bin/env bash
# NVDEC's HEVC decoder and video parser through the bare soname applications link.
#
# nvcuvid_hevc.cpp prints the decoder capabilities, the decoder-creation answers, and, for every
# stream in nvidia/tests/data/hevc, each callback of the video parser (sequence format, picture
# parameters, display order, timestamps) plus the CRC-32 of every displayed surface (NV12, P016 for
# 10-bit streams); nvcuvid_hevc.rtx3060.txt is what the card printed. HEVC decoding is bit-exact by
# definition, so the checksums are the card's.
#
# Not reproduced, and left out of the comparison here (the card's transcript has them):
#   - 4:4:4 HEVC (the card decodes it; VirtualGPU's NVDEC reports it unsupported): the "caps HEVC
#     chroma 3" lines and the "CreateDecoder(HEVC 444" line;
#   - max_display_delay of 2 and 4 on B streams (--extra on the probe): the same pictures in the same
#     order, a few of them one decode callback earlier than on the card.
#
# --card    run against the driver's own library on a real card instead
#           (LD_LIBRARY_PATH from VGPU_CARD_LIBS, default
#           /usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib): the output must still be
#           the golden file, with --update it rewrites the golden file. Card runs include --extra.
# VGPU_HEVC_SKIP=a,b   streams to leave out (sections of the golden file and of the output are not compared); none by default
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
here="$root/nvidia/tests/e2e"
golden="$here/nvcuvid_hevc.rtx3060.txt"
data="$root/nvidia/tests/data"
inc="$root/nvidia/third_party/nvdec_include"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvcuvid_hevc_$$"
mode=sim update=0
skip="${VGPU_HEVC_SKIP:-}"
for a in "$@"; do
  case "$a" in
    --card) mode=card; skip="" ;;
    --update) update=1 ;;
    *) echo "unknown argument $a" >&2; exit 2 ;;
  esac
done
trap 'rm -rf "$out" "$out".txt "$out".err "$out".lib "$out".want "$out".got' EXIT
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
  args+=(--extra)
else
  if [[ ! -e "$shim/libnvcuvid.so.1" ]]; then
    echo "SKIP: libvgpunvcuvid not built (the CUDA ABI headers were absent at build time)"; exit 0
  fi
  # shellcheck disable=SC2207
  flags+=($(shim_sanitizer_nvcc_flags "$shim") -L "$shim")
  env_prefix=(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 "LD_LIBRARY_PATH=$shim" "VGPU_E2E_DATA=$data")
  [[ -n $skip ]] && args+=(--skip "$skip")
fi
nvcc "${flags[@]}" "$here/nvcuvid_hevc.cpp" -o "$out" -ldl -lcuda -lnvcuvid
if [[ $mode == sim ]] && ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
"${env_prefix[@]}" "$out" "${args[@]}" "$data" >"$out.txt" 2>"$out.err" || status=$?
if [[ $mode == card && $update == 1 ]]; then
  cp "$out.txt" "$golden"; echo "wrote $golden"; exit 0
fi
# Leave the skipped streams' sections out of the golden file, and in a simulator run what it does not reproduce.
filter() {
  awk -v skip=",$skip," -v sim="$([[ $mode == sim ]] && echo 1 || echo 0)" '
    /^stream / { name = $2; drop = index(skip, "," name ",") > 0; delay = ($3 == "[delay2]" || $3 == "[delay4]") }
    /^done/ { drop = 0; delay = 0 }
    sim == 1 && (delay || /^caps HEVC chroma 3 / || /^CreateDecoder\(HEVC 444 /) { next }
    !drop { print }' "$1"
}
filter "$golden" >"$out.want"
filter "$out.txt" >"$out.got"
if diff -u "$out.want" "$out.got"; then
  echo "PASS nvcuvid_hevc: $(wc -l <"$out.got") lines identical to the RTX 3060's"
else
  echo "FAIL nvcuvid_hevc: output differs from nvcuvid_hevc.rtx3060.txt"; cat "$out.err"; exit 1
fi
cat "$out.err"
[[ $status == 0 ]] && echo PASS
