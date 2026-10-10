#!/usr/bin/env bash
# The runtime and driver functions the toolkit declares that the shims once lacked (tests/lint/check_header_exports.py),
# called with the argument cases an RTX 3060 was measured on.
#
#   run_exports_sweep.sh runtime|driver                   against the shims
#   run_exports_sweep.sh runtime|driver --card            against NVIDIA's libraries on a real GPU
#   run_exports_sweep.sh runtime|driver --card --update   rewrite the expected file from the card
#
# nvidia/tests/data/exports_sweep_<which>.rtx3060.expected is what the card printed with CUDA 13.0. A case of a
# function newer than the toolkit at hand is not compiled there, and its line (tagged [12.8] or [13]) is left
# out of the expected output. The shim's rtx3060 profile must print the rest unchanged.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
which="${1:-}"
shift || true
case "$which" in
  runtime) src="$root/nvidia/tests/e2e/exports_sweep_runtime.cu" ;;
  driver) src="$root/nvidia/tests/e2e/exports_sweep_driver.cpp" ;;
  *) echo "usage: $0 runtime|driver [--card [--update]]" >&2; exit 2 ;;
esac
module="$root/nvidia/tests/e2e/exports_sweep_module.cu"
expected="$root/nvidia/tests/data/exports_sweep_${which}.rtx3060.expected"
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 runtime|driver [--card [--update]]" >&2; exit 2 ;;
  esac
done
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-exports-sweep.XXXXXX")"
trap 'rm -rf "$work"' EXIT

build() {   # <nvcc> <extra flags...>
  local nvcc_bin="$1"; shift
  "$nvcc_bin" -cubin -arch=sm_86 -Wno-deprecated-gpu-targets "$module" -o "$work/libk.cubin" || return 1
  if [[ $which == runtime ]]; then
    # The runtime shim hands the library calls to the driver shim, found in the process: link libcuda whether or not
    # the program names one of its functions.
    "$nvcc_bin" -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets -w "$@" "$src" -o "$work/prog" \
      -Xlinker --no-as-needed -lcuda
  else
    "$nvcc_bin" -std=c++17 -x c++ -cudart none -Wno-deprecated-gpu-targets -w "$@" "$src" -o "$work/prog" -lcuda
  fi
}

# The lines of the expected file this toolkit's program prints.
release_of() { "$1" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p'; }
filter_expected() {   # <major> <minor>
  local major=$1 minor=$2 f="$work/expected"
  cp "$expected" "$f"
  if (( major < 13 )); then grep -v ' \[13\]' "$f" > "$f.2" && mv "$f.2" "$f"; fi
  if (( major * 100 + minor < 1208 )); then grep -v ' \[12.8\]' "$f" > "$f.2" && mv "$f.2" "$f"; fi
  # the trailer counts the cases
  local n; n="$(grep -c . "$f")"
  sed -i '$d' "$f"
  echo "PASS: $((n - 1)) cases" >> "$f"
}

if (( card )); then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64"; do
    [[ -e "$d/libcudart.so" || -e "$d"/libcudart.so.[0-9]* ]] && libs="$d" && break
  done
  [[ -n "$libs" && -e "$libs/stubs/libcuda.so" ]] || { echo "SKIP: no CUDA libraries beside nvcc"; exit 0; }
  build "$(command -v nvcc)" -L"$libs/stubs" || { echo "FAIL: does not compile"; exit 1; }
  (cd "$work" && LD_LIBRARY_PATH="$libs" ./prog > out.txt) || { echo "FAIL: the program failed on the card"; cat "$work/out.txt"; exit 1; }
  if (( update )); then cp "$work/out.txt" "$expected"; echo "wrote $expected"; exit 0; fi
  read -r major minor <<< "$(release_of "$(command -v nvcc)")"
  filter_expected "$major" "$minor"
  diff -u "$work/expected" "$work/out.txt" && echo "the $which functions match the card's" || { echo "FAIL: differs from what the card printed"; exit 1; }
  exit 0
fi

shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
build "$nvcc_bin" "${san_flags[@]}" -L"$shim" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$work/prog"; then exit 0; fi
read -r major minor <<< "$(release_of "$nvcc_bin")"
filter_expected "$major" "$minor"
(cd "$work" && env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" ./prog > out.txt 2> err.txt) \
  || { echo "FAIL: the program failed"; cat "$work/err.txt"; diff -u "$work/expected" "$work/out.txt"; exit 1; }
diff -u "$work/expected" "$work/out.txt" || { echo "FAIL: the rtx3060 profile does not print what the card printed"; exit 1; }
echo "PASS"
