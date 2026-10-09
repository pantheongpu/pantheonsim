#!/usr/bin/env bash
# NVDEC through the bare soname applications link: libvgpunvcuvid in the shim
# directory, standing in for the driver's libnvcuvid.so.1.
#
# nvcuvid_paths.cpp prints one line per fact about the cuvid API and compares the
# decoded NV12 pixels of the JPEG fixtures with the card's
# (nvidia/tests/data/nvdec/*.nv12); nvcuvid_paths.rtx3060.txt is what the card
# printed.
#
# --card    run against the driver's own library on a real card instead
#           (LD_LIBRARY_PATH from VGPU_CARD_LIBS, default
#           /usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib): the output must still be
#           the golden file, the pixels the data files, to within one level
# --update  with --card, rewrite the golden file and the pixel files from the card
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
here="$root/nvidia/tests/e2e"
golden="$here/nvcuvid_paths.rtx3060.txt"
data="$root/nvidia/tests/data"
inc="$root/nvidia/third_party/nvdec_include"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvcuvid_$$"
mode=sim update=0
for a in "$@"; do
  case "$a" in
    --card) mode=card ;;
    --update) update=1 ;;
    *) echo "unknown argument $a" >&2; exit 2 ;;
  esac
done
trap 'rm -rf "$out" "$out".txt "$out".err "$out".lib' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
flags=(-w -std=c++17 -cudart none -arch=compute_86 -code=compute_86 -Wno-deprecated-gpu-targets -I "$inc")
args=()
if [[ $mode == card ]]; then
  libdir="${VGPU_CARD_LIBS:-/usr/local/cuda-13.0/lib64:/usr/lib/wsl/lib}"
  # The driver ships libnvcuvid.so.1 only; link against it by a name the linker finds.
  mkdir -p "$out.lib"
  for d in ${libdir//:/ }; do
    [[ -e "$d/libnvcuvid.so.1" ]] && ln -sf "$d/libnvcuvid.so.1" "$out.lib/libnvcuvid.so" && break
  done
  [[ -e "$out.lib/libnvcuvid.so" ]] || { echo "SKIP: no libnvcuvid.so.1 in $libdir"; exit 0; }
  flags+=(-L "$out.lib" -L /usr/lib/wsl/lib)
  env_prefix=(env "LD_LIBRARY_PATH=$libdir" "VGPU_E2E_DATA=$data")
  if [[ $update == 1 ]]; then args=(--update "$data/nvdec"); mkdir -p "$data/nvdec"; else args=(--tolerance 1); fi
else
  if [[ ! -e "$shim/libnvcuvid.so.1" ]]; then
    echo "SKIP: libvgpunvcuvid not built (the CUDA ABI headers were absent at build time)"; exit 0
  fi
  # shellcheck disable=SC2207
  flags+=($(shim_sanitizer_nvcc_flags "$shim") -L "$shim")
  env_prefix=(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 "LD_LIBRARY_PATH=$shim" "VGPU_E2E_DATA=$data")
fi
nvcc "${flags[@]}" "$here/nvcuvid_paths.cpp" -o "$out" -ldl -lcuda -lnvcuvid
if [[ $mode == sim ]] && ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
"${env_prefix[@]}" "$out" "${args[@]}" >"$out.txt" 2>"$out.err" || status=$?
if [[ $mode == card && $update == 1 ]]; then
  cp "$out.txt" "$golden"; echo "wrote $golden and $data/nvdec"; exit 0
fi
if diff -u "$golden" "$out.txt"; then
  echo "PASS nvcuvid_paths: $(wc -l <"$out.txt") lines identical to the RTX 3060's"
else
  echo "FAIL nvcuvid_paths: output differs from nvcuvid_paths.rtx3060.txt"; cat "$out.err"; exit 1
fi
cat "$out.err"
[[ $status == 0 ]] && echo PASS
