#!/usr/bin/env bash
# An unmodified NVIDIA profiler, profiling a program that runs on the simulator.
#
# This is the test the hand-written CUPTI consumer cannot be: nvprof is a tool
# nobody here wrote, and it only collects anything if the whole contract holds --
# the injection library gets loaded and initialized at CUDA startup, the CUPTI
# soname and symbol version match what it links against, every symbol it needs
# resolves, and the activity records describe real work.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvprof_$$"

command -v nvprof >/dev/null 2>&1 || { echo "SKIP: nvprof not installed"; exit 0; }
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
shopt -s nullglob
cupti_libs=("$shim"/libcupti.so.[0-9]*)
shopt -u nullglob
(( ${#cupti_libs[@]} )) || { echo "SKIP: libvgpucupti not built"; exit 0; }

# nvprof links a CUPTI of its own toolkit's major, and loads whichever one the
# library path offers. A CUDA 12 nvprof against a CUDA 13 shim finds no
# libcupti.so.12, falls back to the real one, and reports collecting nothing --
# an unbuildable pairing rather than a defect, so it skips like the rest.
# nvprof cannot be run against a sanitizer-instrumented shim, and the reason is
# nvprof rather than anything here: it loads libcuda into its *own* process to
# look for devices, so with the shim on the library path an uninstrumented
# NVIDIA binary ends up loading an instrumented library and aborts with "ASan
# runtime does not come first in initial library list". It does this even
# profiling /bin/true, so building the app with the shim's sanitizer -- which is
# what every other e2e test here does, and the obvious thing to try -- does not
# help. nvprof is not ours to rebuild.
#
# Nothing is lost by skipping: what the sanitizer builds check is this project's
# own memory safety, and e2e_cupti_activity exercises the same injection and
# activity-record path under the sanitizers with a program we do control. What
# is unique to this test -- that a profiler nobody here wrote collects real data
# -- is covered by the ordinary build.
san="$(shim_sanitizer "$shim")"
if [[ -n "$san" ]]; then
  echo "SKIP: shim is built with $san, and nvprof loads libcuda into its own"
  echo "      uninstrumented process (see the comment in this script)"
  exit 0
fi

shim_major="${cupti_libs[0]##*.}"
nvprof_major="$(nvprof --version 2>&1 | sed -n 's/.*Release version \([0-9]*\)\..*/\1/p' | head -1)"
if [[ -n "$nvprof_major" && "$shim_major" != "$nvprof_major" ]]; then
  echo "SKIP: nvprof is CUDA $nvprof_major but the shim is CUDA $shim_major"
  exit 0
fi

# nvprof refuses compute capability 8.0 and above -- its own deprecation, not
# ours -- so this profiles the Turing profile, which it does support. The
# program has to be built for that architecture too: PTX runs forward onto
# newer hardware, never backward onto older.
# The app has to carry the same toolkit major as the shim, and a machine may
# have more than one nvcc. Pick whichever matches rather than skipping because
# the first one on PATH happens to be the other.
pick_nvcc=""
for cand in "${VGPU_NVCC:-}" "$(command -v nvcc 2>/dev/null)" /usr/bin/nvcc /usr/local/cuda/bin/nvcc; do
  [[ -n "$cand" && -x "$cand" ]] || continue
  cand_major="$("$cand" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)[,.].*/\1/p' | head -1)"
  [[ "$cand_major" == "$shim_major" ]] && { pick_nvcc="$cand"; break; }
done
[[ -n "$pick_nvcc" ]] || { echo "SKIP: no nvcc for CUDA $shim_major"; exit 0; }

nvcc_host_compiler_fix

"$pick_nvcc" -std=c++14 -cudart shared -arch=compute_75 -code=compute_75 \
     -Wno-deprecated-gpu-targets "$root/tests/e2e/vector_add.cu" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi

log="$out.log"
VGPU_QUIET=1 VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" nvprof "$out" > "$log" 2>&1 || true
rm -f "$out"

if ! grep -q "PASS" "$log"; then
  echo "FAIL: the program itself did not pass under nvprof"; sed 's/^/    /' "$log"; rm -f "$log"; exit 1
fi
if grep -q "No profile data collected" "$log"; then
  echo "FAIL: nvprof collected nothing"; sed 's/^/    /' "$log"; rm -f "$log"; exit 1
fi
for want in "vecAdd" "CUDA memcpy HtoD" "CUDA memcpy DtoH"; do
  if ! grep -q "$want" "$log"; then
    echo "FAIL: nvprof output does not mention '$want'"; sed 's/^/    /' "$log"; rm -f "$log"; exit 1
  fi
done
rm -f "$log"
echo "nvprof profiled the simulator: kernel and both copy directions reported"
