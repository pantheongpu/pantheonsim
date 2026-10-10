#!/usr/bin/env bash
# NVDEC's MPEG-2 decoder and video parser through the bare soname applications link.
#
# nvcuvid_mpeg2.cpp prints the decoder capabilities, the decoder-creation answers, and, for every
# stream in nvidia/tests/data/mpeg2, each callback of the video parser (sequence format, picture
# parameters, display order, timestamps) and whether every displayed frame is within bounds of the
# card's (the pixel files in nvidia/tests/data/nvdec/mpeg2: MPEG-2 leaves the inverse DCT to the
# decoder, so VirtualGPU's frames differ from the card's by a level in a few percent of the samples);
# nvcuvid_mpeg2.rtx3060.txt is what the card printed.
#
# Not reproduced, and left out of the comparison here (the card's transcript has them):
#   - MPEG-1 (the card decodes it; VirtualGPU's NVDEC reports it unsupported): the "caps MPEG1"
#     lines and the stream hdr_no_extension (a sequence header without extension);
#   - max_display_delay of 2 on the stream with field pictures (fields_ipb_tff [delay2]): the same
#     pictures in the same order, some of them one decode callback earlier or later than on the card;
#   - max_display_delay of 3 and more on streams with B pictures (--extra on the probe): the same
#     pictures in the same order, a few of them one decode callback later on the card than here.
#
# --card    run against the driver's own library on a real card instead
#           (LD_LIBRARY_PATH from VGPU_CARD_LIBS, default
#           /usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib): the output must still be
#           the golden file; with --update it rewrites the golden file and the pixel files.
#           Card runs include --extra.
# VGPU_MPEG2_SKIP=a,b   streams to leave out (sections of the golden file and of the output are not compared); none by default
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
here="$root/nvidia/tests/e2e"
golden="$here/nvcuvid_mpeg2.rtx3060.txt"
data="$root/nvidia/tests/data"
inc="$root/nvidia/third_party/nvdec_include"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvcuvid_mpeg2_$$"
mode=sim update=0
skip="${VGPU_MPEG2_SKIP:-}"
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
  args+=(--extra --golden "$data/nvdec/mpeg2")
  if [[ $update == 1 ]]; then mkdir -p "$data/nvdec/mpeg2"; args+=(--update); fi
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
  args+=(--golden "$data/nvdec/mpeg2")
  [[ -n $skip ]] && args+=(--skip "$skip")
fi
nvcc "${flags[@]}" "$here/nvcuvid_mpeg2.cpp" -o "$out" -ldl -lcuda -lnvcuvid
if [[ $mode == sim ]] && ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
"${env_prefix[@]}" "$out" "${args[@]}" "$data" >"$out.txt" 2>"$out.err" || status=$?
if [[ $mode == card && $update == 1 ]]; then
  cp "$out.txt" "$golden"; echo "wrote $golden"; exit 0
fi
# Leave the skipped streams' sections out of the golden file, and in a simulator run what it does not reproduce.
filter() {
  awk -v skip=",$skip," -v sim="$([[ $mode == sim ]] && echo 1 || echo 0)" '
    /^stream / { name = $2; drop = index(skip, "," name ",") > 0; delay = ($3 == "[delay3]" || $3 == "[delay4]" || (name == "fields_ipb_tff" && $3 == "[delay2]") || name == "hdr_no_extension") }
    /^done/ { drop = 0; delay = 0 }
    sim == 1 && (delay || /^caps MPEG1 /) { next }
    !drop { print }' "$1"
}
filter "$golden" >"$out.want"
filter "$out.txt" >"$out.got"
if diff -u "$out.want" "$out.got"; then
  echo "PASS nvcuvid_mpeg2: $(wc -l <"$out.got") lines identical to the RTX 3060's"
else
  echo "FAIL nvcuvid_mpeg2: output differs from nvcuvid_mpeg2.rtx3060.txt"; cat "$out.err"; exit 1
fi
cat "$out.err"
[[ $status == 0 ]] && echo PASS
