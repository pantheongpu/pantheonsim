#!/usr/bin/env bash
# Host-GPU mode (docs/perf-design/09-host-gpu-spike.md): VGPU_HOST_GPU=exact runs launches whose kernels are in the exact set on the machine's
# real GPU, and the program has to print what it prints with the mode off. Needs a card of the model the simulator is told to be, so it
# SKIPs without one (no hosted CI machine has one). host_gpu_exact.cu runs kernels of every kind: integer and FFMA kernels (on the card),
# an expf kernel (MUFU: stays in the simulator), a tiny launch (too small), a kernel with a module variable (refused).
#   1. mode off and mode exact print the same lines;
#   2. the exact run reports launches on the card, and a kernel outside the set, a launch too small and a refused launch;
#   3. the card copy outlives a launch (the default; VGPU_HOST_GPU_WRITE_THROUGH=1 is the older mode): the program touches memory between launches in every
#      way the host can (partial write, memset, device copy, free and reallocate, a launch of the simulator's in between) and the lines it prints have to be
#      the same with the mode on, in either form, and off; and the persistent run has to move less than the write-through one;
#   4. VGPU_HOST_GPU_VERIFY=1 (every launch the mode would offload also runs in the simulator, and the memory is compared) finds every
#      kernel of the exact set identical and ran at least two on the card.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
command -v nvidia-smi >/dev/null 2>&1 || { echo "SKIP: no nvidia-smi: no GPU on this machine"; exit 0; }
name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')"
[[ -n "$name" && -n "$cc" ]] || { echo "SKIP: nvidia-smi reports no GPU"; exit 0; }
# The simulated profile has to be the card: "NVIDIA A10G" is nvidia/a10g, "NVIDIA GeForce RTX 3060" is nvidia/rtx3060.
id="$(echo "$name" | tr 'A-Z' 'a-z' | sed -e 's/^nvidia //' -e 's/^geforce //' -e 's/ //g')"
profile="${VGPU_HOST_GPU_TEST_PROFILE:-nvidia/$id}"
[[ -f "$root/nvidia/profiles/${profile#nvidia/}.yaml" ]] || { echo "SKIP: no simulator profile for the card ($name -> $profile)"; exit 0; }
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-host-gpu.XXXXXX")"
trap 'rm -rf "$work"' EXIT
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/host_gpu_exact.cu" "$root/nvidia/tests/e2e/host_gpu_exact_mod.cu" -o "$work/prog" \
  || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$work/prog"; then exit 0; fi
run() {   # <stdout file> <stderr file> [env...]
  local out=$1 err=$2
  shift 2
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$@" "$work/prog" > "$out" 2> "$err"
}
fails=0
run "$work/off.out" "$work/off.err" || { echo "FAIL: the run with the mode off failed"; tail -5 "$work/off.err"; exit 1; }
run "$work/exact.out" "$work/exact.err" VGPU_HOST_GPU=exact || { echo "FAIL: the run with the mode on failed"; tail -5 "$work/exact.err"; fails=1; }
if grep -q "mode off" "$work/exact.err"; then
  echo "SKIP: the mode did not start on this machine: $(grep 'mode off' "$work/exact.err" | head -1)"; exit 0
fi
diff -u "$work/off.out" "$work/exact.out" > "$work/d.txt" || { echo "FAIL: the mode changes the program's output:"; head -20 "$work/d.txt"; fails=1; }
sum="$(grep '^host_gpu: [0-9]* launches ran on the card' "$work/exact.err" | tail -1)"
[[ -n "$sum" ]] || { echo "FAIL: no host_gpu summary line"; fails=1; }
onc="$(echo "$sum" | sed -n 's/^host_gpu: \([0-9]*\) launches ran.*/\1/p')"
notset="$(echo "$sum" | sed -n 's/.* (\([0-9]*\) kernels not in the exact set.*/\1/p')"
small="$(echo "$sum" | sed -n 's/.*set, \([0-9]*\) too small.*/\1/p')"
shape="$(echo "$sum" | sed -n 's/.*small, \([0-9]*\) unsupported.*/\1/p')"
(( ${onc:-0} >= 8 )) || { echo "FAIL: fewer than 8 launches ran on the card: $sum"; fails=1; }
(( ${notset:-0} >= 1 )) || { echo "FAIL: the expf kernel was not refused as outside the exact set: $sum"; fails=1; }
(( ${small:-0} >= 1 )) || { echo "FAIL: the one-warp launch was not refused as too small: $sum"; fails=1; }
(( ${shape:-0} >= 1 )) || { echo "FAIL: the module-variable kernel was not refused: $sum"; fails=1; }
run "$work/wt.out" "$work/wt.err" VGPU_HOST_GPU=exact VGPU_HOST_GPU_WRITE_THROUGH=1 || { echo "FAIL: the write-through run failed"; tail -5 "$work/wt.err"; fails=1; }
diff -u "$work/off.out" "$work/wt.out" > "$work/d.txt" || { echo "FAIL: write-through changes the program's output:"; head -20 "$work/d.txt"; fails=1; }
moved() { sed -n 's/^host_gpu: \([0-9.]*\) MiB went to the card, \([0-9.]*\) MiB came back.*/\1 \2/p' "$1" | tail -1; }
read -r p_in p_out <<< "$(moved "$work/exact.err")"
read -r w_in w_out <<< "$(moved "$work/wt.err")"
awk -v a="${p_in:-0}" -v b="${p_out:-0}" -v c="${w_in:-1}" -v d="${w_out:-1}" 'BEGIN { exit !(a + b < 0.5 * (c + d)) }' \
  || { echo "FAIL: the persistent run moved ${p_in:-?}+${p_out:-?} MiB, not under half of write-through's ${w_in:-?}+${w_out:-?}"; fails=1; }
# Verification: the simulator and the card run the same launches; the memory has to be identical for every kernel in the exact set.
run "$work/v.out" "$work/v.err" VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 || { echo "FAIL: the verify run failed"; tail -5 "$work/v.err"; fails=1; }
vline="$(grep '^host_gpu: verify: kernels in the exact set' "$work/v.err" | tail -1)"
same="$(echo "$vline" | sed -n 's/.*exact set: \([0-9]*\) identical.*/\1/p')"
diffc="$(echo "$vline" | sed -n 's/.*identical, \([0-9]*\) different;.*/\1/p')"
(( ${same:-0} >= 2 )) || { echo "FAIL: verify compared fewer than 2 kernels of the exact set: $vline"; fails=1; }
(( ${diffc:-1} == 0 )) || { echo "FAIL: a kernel of the exact set gave other bits on the card: $vline"; grep DIFFERENT "$work/v.err" | head; fails=1; }
cmp -s "$work/off.out" "$work/v.out" || { echo "FAIL: the verify run changed the program's output"; fails=1; }
(( fails == 0 )) && { echo "PASS: $sum"; exit 0; }
exit 1
