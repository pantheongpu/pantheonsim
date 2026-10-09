#!/usr/bin/env bash
# A program that traces itself through CUPTI, run against the shims and compared
# with what NVIDIA's own libcupti printed for the same program on an RTX 3060.
#
#   run_cupti_case.sh <case>                 run against the shims and compare
#                                            with nvidia/tests/data/cupti_<case>.expected
#   run_cupti_case.sh <case> --card          run against NVIDIA's libraries on a
#                                            real GPU and compare with the same
#                                            file: the check that the expected
#                                            output is what hardware prints
#   run_cupti_case.sh <case> --card --update rewrite the expected file from the card
#
# Cases (nvidia/tests/e2e/cupti_<case>.cu):
#   trace    runtime-API callbacks, resource and synchronize callbacks, and the
#            kernel, copy, fill, wait, stream, device and runtime records
#   nvtx     NVTX markers, ranges and domains (callbacks and MARKER records)
#   extcorr  external correlation ids pushed around runtime calls
#   params   the parameter structure of ~100 runtime calls, field by field
#   memory   allocation, release and pool records, and the older memory kind
#   graph    graph ids, graph-trace records, and the resource callbacks of graphs
#   resource module, stream-attribute and context callbacks; context, stream and
#            function records
#   buffers  when the program is asked for a buffer: as the first record of a
#            batch is made, not when it is delivered
#   um       Unified Memory counters: configuring them, and (where the driver pages
#            managed memory on demand) what a managed allocation produces
#   peer     copies between two devices with a peer path (SKIPped where there is none)
#   misc     which kinds can be enabled, per-function records, callback switches,
#            CUDA event records, copies between devices, overhead, a program clock
#            (needs two GPUs on the card)
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
case_name="${1:?usage: $0 <trace|nvtx|extcorr|params|memory|graph|resource|misc|overhead|filter|filter_driver|buffers|um|peer> [--card [--update]]}"
shift
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cupti_${case_name}.cu"
expected="$root/nvidia/tests/data/cupti_${case_name}.expected"
out="${TMPDIR:-/tmp}/vgpu_e2e_cupti_${case_name}_$$"
[[ -f "$src" ]] || { echo "no such case: $case_name" >&2; exit 2; }
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 <case> [--card [--update]]" >&2; exit 2 ;;
  esac
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ "$case_name" == nvtx ]]; then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  if ! find "$cuda_root/include" "$cuda_root/targets/x86_64-linux/include" /usr/include -maxdepth 2 -name nvToolsExt.h 2>/dev/null | grep -q .; then
    echo "SKIP: NVTX headers not found"; exit 0
  fi
fi
trap 'rm -f "$out" "$out.txt" "$out.cubin"' EXIT

# The expected files are what a CUDA 13.0 libcupti printed. A case whose program
# leaves out what an older toolkit's cupti.h cannot name (it is built under
# CUPTI_API_VERSION guards, as the trace's records and callbacks are) prints a
# different, shorter trace there, which says nothing about the shim: SKIP below
# the CUPTI_API_VERSION that the case needs for its whole trace to be asked for.
#   graph     executable-graph ids (22: CUDA 12.4)
#   resource  the stream-attribute callback (22)
#   filter, filter_driver  the per-function activity switches (24: CUDA 12.5)
#   misc      CUDA event records, device timestamps and the per-function switches
#             of the newer records (13.0)
case "$case_name" in
  graph|resource) need_api=22 ;;
  filter|filter_driver) need_api=24 ;;
  misc) need_api=130000 ;;
  *) need_api=0 ;;
esac
if (( need_api > 0 )); then
  nvcc_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  api_header=""
  for d in "$nvcc_root/include" "$nvcc_root/targets/x86_64-linux/include" /usr/include; do
    [[ -f "$d/cupti_version.h" ]] && api_header="$d/cupti_version.h" && break
  done
  have_api=0
  [[ -n "$api_header" ]] && have_api="$(sed -n 's/^#define CUPTI_API_VERSION[[:space:]]\+\([0-9]\+\).*/\1/p' "$api_header" | head -1)"
  have_api="${have_api:-0}"
  if (( have_api < need_api )); then
    echo "SKIP: this toolkit's CUPTI (API version $have_api) is older than the $need_api the $case_name case needs"; exit 0
  fi
fi

# The resource case loads a cubin through the driver API; nvcc makes it for the
# device both runs use, so the card and the shim are handed the same bytes.
cubin=""
devices=1
[[ "$case_name" == misc || "$case_name" == peer ]] && devices=2
if [[ "$case_name" == resource ]]; then
  cubin="$out.cubin"
  nvcc -cubin -arch=sm_86 -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/cupti_resource_module.cu" -o "$cubin"
fi

if (( card )); then
  # NVIDIA's libraries: the toolkit's own libcupti and the driver's.
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64" /usr/lib/x86_64-linux-gnu; do
    compgen -G "$d/libcupti.so*" >/dev/null && libs="$d" && break
  done
  [[ -n "$libs" ]] || { echo "SKIP: no libcupti beside nvcc"; exit 0; }
  stubs="$libs/stubs"
  [[ -e "$stubs/libcuda.so" ]] || { echo "SKIP: no libcuda stub beside nvcc"; exit 0; }
  if [[ "$case_name" == misc || "$case_name" == peer ]] && command -v nvidia-smi >/dev/null 2>&1 &&
     (( $(nvidia-smi -L 2>/dev/null | grep -c GPU) < 2 )); then
    echo "SKIP: the misc case's expected output was made on two GPUs"; exit 0
  fi
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" -lcupti -L"$libs" -lcuda -L"$stubs"
  # NVTX reaches a tool through the library named here, as a profiler sets it.
  CUPTI_TEST_CUBIN="$cubin" NVTX_INJECTION64_PATH="$(ls "$libs"/libcupti.so.[0-9]* | head -1)" \
      LD_LIBRARY_PATH="$libs" "$out" > "$out.txt"
else
  shopt -s nullglob
  cupti_libs=("$shim"/libcupti.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cupti_libs[@]} == 0 )); then
    echo "SKIP: libvgpucupti not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  if [[ "$case_name" == nvtx ]] && ! nm -D "${cupti_libs[0]}" 2>/dev/null | grep InitializeInjectionNvtx2 >/dev/null; then
    echo "SKIP: libvgpucupti built without NVTX (no nvtx3 headers at build time)"; exit 0
  fi
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcupti -lcuda -L"$shim"
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  # The RTX 3060 profile: the trace names the device it ran on. Both shims in one
  # process (libcuda and libcudart) need the sanitizer builds told so.
  env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT="$devices" CUPTI_TEST_CUBIN="$cubin" \
      NVTX_INJECTION64_PATH="${cupti_libs[0]}" LD_LIBRARY_PATH="$shim" "$out" > "$out.txt"
fi

if (( card )) && [[ "$case_name" == peer ]] && head -1 "$out.txt" | grep -q ": no$"; then
  echo "SKIP: no peer path between this machine's first two GPUs"; exit 0
fi
if (( card && update )); then
  cp "$out.txt" "$expected"
  echo "wrote $expected"
  exit 0
fi
if diff -u "$expected" "$out.txt"; then
  echo "CUPTI $case_name trace matches the card's"
else
  echo "FAIL: the $case_name trace differs from what NVIDIA's CUPTI printed for the same program"
  exit 1
fi
