#!/usr/bin/env bash
# The host-GPU mode's start-up (include/vgpu/host_gpu.hpp start_worthwhile; src/runtime/host_gpu.cpp begin_start / await_start), tested with a stand-in driver
# (host_gpu_stub_driver.c) so that it runs on every machine, card or not. What it pins:
#   1. a program with no launch worth a card (a kernel outside the exact set, or a launch of a few warps) never loads the driver: no call reaches it;
#   2. a launch of a kernel in the exact set on a grid worth a card starts the driver, and the simulator does not wait for it: the first launch takes far
#      less than the driver's start (the stub sleeps STUB_INIT_MS in cuInit, as the real driver's context creation takes ~150 ms);
#   3. the card path is reached later (the stub's module load is called), and the program's output is the one it prints with the mode off;
#   4. a driver that cannot start says "mode off" once, the program's output is unchanged and no later launch calls the driver again.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
profile="${VGPU_HOST_GPU_TEST_PROFILE:-nvidia/rtx3060}"
yaml="$root/nvidia/profiles/${profile#nvidia/}.yaml"
[[ -f "$yaml" ]] || { echo "SKIP: no profile $profile"; exit 0; }
cc="$(sed -n 's/^compute_capability: "\([0-9]*\)\.\([0-9]*\)".*/\1\2/p' "$yaml" | head -1)"
mp="$(sed -n 's/^  multiprocessors: \([0-9]*\).*/\1/p' "$yaml" | head -1)"
[[ -n "$cc" && -n "$mp" ]] || { echo "FAIL: cannot read the compute capability and SM count of $yaml"; exit 1; }
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
cc_bin="$(command -v cc || command -v gcc)"
[[ -n "$cc_bin" ]] || { echo "SKIP: no C compiler"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-host-gpu-stub.XXXXXX")"
trap 'rm -rf "$work"' EXIT
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/host_gpu_lazy.cu" -o "$work/prog" || { echo "FAIL: does not compile"; exit 1; }
"$cc_bin" -shared -fPIC -O1 -o "$work/libstubcuda.so" "$root/nvidia/tests/e2e/host_gpu_stub_driver.c" || { echo "FAIL: the stand-in driver does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$work/prog"; then exit 0; fi
init_ms=2000
fails=0
run() {   # <tag> <program argument> [env...]: stdout in $work/<tag>.out, stderr in $work/<tag>.err, driver calls in $work/<tag>.log
  local tag=$1 arg=$2
  shift 2
  : > "$work/$tag.log"
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" STUB_LOG="$work/$tag.log" STUB_SM="$cc" STUB_MP="$mp" STUB_INIT_MS=$init_ms \
      VGPU_HOST_GPU_LIBCUDA="$work/libstubcuda.so" "$@" "$work/prog" "$arg" > "$work/$tag.out" 2> "$work/$tag.err"
}
calls() { sort "$work/$1.log" | uniq -c | tr -s ' ' | tr '\n' ';'; }
for arg in big small mufu; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; tail -5 "$work/off_$arg.err"; exit 1; }
done
# 1. nothing worth a card: the driver is never touched.
for arg in small mufu; do
  run "on_$arg" "$arg" VGPU_HOST_GPU=exact || { echo "FAIL: the $arg run with the mode on failed"; tail -5 "$work/on_$arg.err"; fails=1; }
  [[ ! -s "$work/on_$arg.log" ]] || { echo "FAIL: the $arg program (nothing worth a card) reached the driver: $(calls "on_$arg")"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/on_$arg.out" > /dev/null || { echo "FAIL: the mode changes the output of $arg"; fails=1; }
  grep -q "mode off" "$work/on_$arg.err" && { echo "FAIL: $arg: the mode said it was off: $(grep 'mode off' "$work/on_$arg.err" | head -1)"; fails=1; }
done
# 2 and 3. a launch worth a card: the driver starts, and the simulator does not wait for it.
run on_big big VGPU_HOST_GPU=exact || { echo "FAIL: the big run with the mode on failed"; tail -5 "$work/on_big.err"; fails=1; }
diff -q "$work/off_big.out" "$work/on_big.out" > /dev/null || { echo "FAIL: the mode changes the output of the big program"; fails=1; }
grep -q '^cuInit$' "$work/on_big.log" || { echo "FAIL: a launch worth a card did not start the driver: $(calls on_big)"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/on_big.log" || { echo "FAIL: no launch reached the card path (the stand-in's module load): $(calls on_big)"; fails=1; }
first="$(sed -n 's/^first_launch_ms=\([0-9]*\).*/\1/p' "$work/on_big.err" | head -1)"
[[ -n "$first" ]] || { echo "FAIL: the program printed no first_launch_ms"; fails=1; first=0; }
(( first < init_ms / 2 )) || { echo "FAIL: the first launch took $first ms: it waited for a driver start of $init_ms ms"; fails=1; }
# 4. a driver that cannot start.
run on_bad big VGPU_HOST_GPU=exact STUB_FAIL_INIT=1 || { echo "FAIL: the run with a failing driver failed"; tail -5 "$work/on_bad.err"; fails=1; }
diff -q "$work/off_big.out" "$work/on_bad.out" > /dev/null || { echo "FAIL: a driver that cannot start changes the output"; fails=1; }
n="$(grep -c 'cuInit failed; mode off' "$work/on_bad.err")"
(( n == 1 )) || { echo "FAIL: expected exactly one 'cuInit failed; mode off' line, got $n"; fails=1; }
! grep -q '^cuModuleLoadData$' "$work/on_bad.log" || { echo "FAIL: a launch reached the card after the driver failed to start"; fails=1; }
(( fails == 0 )) && { echo "PASS: first launch $first ms against a driver start of $init_ms ms; driver calls in the big run: $(calls on_big)"; exit 0; }
exit 1
