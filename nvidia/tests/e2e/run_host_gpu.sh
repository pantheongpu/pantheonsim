#!/usr/bin/env bash
# Host-GPU mode (docs/perf-design/09-host-gpu-spike.md): VGPU_HOST_GPU=exact runs launches whose kernels are in the exact set on the machine's
# real GPU, and the program has to print what it prints with the mode off. Needs a card of the model the simulator is told to be, so it
# SKIPs without one (no hosted CI machine has one). host_gpu_exact.cu runs kernels of every kind: integer and FFMA kernels (on the card),
# an expf kernel (MUFU: stays in the simulator), a tiny launch (too small), a kernel that calls assert (refused).
#   1. mode off and mode exact print the same lines;
#   2. the exact run reports launches on the card, and a kernel outside the set, a launch too small and a refused launch;
#   3. the card copy outlives a launch (the default; VGPU_HOST_GPU_WRITE_THROUGH=1 is the older mode): the program touches memory between launches in every
#      way the host can (partial write, memset, device copy, free and reallocate, a launch of the simulator's in between) and the lines it prints have to be
#      the same with the mode on, in either form, and off; and the persistent run has to move less than the write-through one;
#   4. VGPU_HOST_GPU_VERIFY=1 (every launch the mode would offload also runs in the simulator, and the memory is compared) finds every
#      kernel of the exact set identical and ran at least two on the card.
# Turbo mode (VGPU_HOST_GPU=turbo, docs/perf-design/12-host-gpu-turbo.md):
#   5. the expf kernel, which exact mode keeps in the simulator, runs on the card; the lines of the program that do not hold expf results are the ones the
#      mode-off run prints, and exact mode's summary still counts the expf launches as outside its set (exact is unchanged);
#   6. VGPU_HOST_GPU_VERIFY=1 in turbo mode compares by tolerance: the expf kernel ran on the card, no word is outside the default tolerance, and the
#      program's output is the mode-off output to the byte (under verify the simulator's result is the program's).
# What a module holds besides code (Round sp66, host_gpu_vars.cu):
#   7. kernels that use __device__ and __constant__ variables (by name, through a pointer from cudaGetSymbolAddress, with cudaMemcpyToSymbol before every
#      launch) run on the card in both modes and the output is the mode-off output to the byte; exact mode's verify finds them identical;
#   8. in turbo mode a kernel that prints runs on the card, and the lines (the card's and the host's, stdout a file) come in the mode-off order;
#   9. with a live texture object a plain kernel runs on the card in both modes, and the output is the mode-off output;
#  10. in turbo mode a kernel that calls a device function of another section (host_gpu_rdc.cu, -rdc=true) runs on the card, output the mode-off output;
#  11. a kernel launched from a CUDA graph runs on the card in both modes, output the mode-off output.
# The idioms of exact mode (Round sp68):
#  12. with VGPU_HOST_GPU_SQRT=1 (and VGPU_HOST_GPU_DIV=1) the sqrtf kernel and the sqrtf-then-division kernel of host_gpu_lazy.cu (inputs on both paths of
#      sqrtf, negatives among them) run on the card, verify finds every launch identical, and the output is the mode-off output to the byte.
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
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/host_gpu_vars.cu" -o "$work/vars" \
  || { echo "FAIL: host_gpu_vars.cu does not compile"; exit 1; }
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -rdc=true -cudart shared -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/host_gpu_rdc.cu" "$root/nvidia/tests/e2e/host_gpu_rdc_fn.cu" -o "$work/rdc" \
  || { echo "FAIL: host_gpu_rdc.cu does not compile"; exit 1; }
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
(( ${shape:-0} >= 1 )) || { echo "FAIL: the kernel that calls assert was not refused: $sum"; fails=1; }
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
# 5. turbo mode: the expf kernel goes to the card. Its results are the card's (one or two ulp from the simulator's), so the fields that hash them are
# left out of the comparison: "expf <hash>" of the pass lines and the first hash of the "fma after simulator" lines (db, which use_expf wrote).
no_expf() { sed -e 's/ expf [0-9a-f]\{16\}/ expf -/' -e 's/\(fma after simulator  \)[0-9a-f]\{16\}/\1-/' "$1"; }
run "$work/rel.out" "$work/rel.err" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the turbo run failed"; tail -5 "$work/rel.err"; fails=1; }
diff -u <(no_expf "$work/off.out") <(no_expf "$work/rel.out") > "$work/d.txt" || { echo "FAIL: turbo mode changes output that does not come from expf:"; head -20 "$work/d.txt"; fails=1; }
grep -q 'use_expf[^ ]* on the card' "$work/rel.err" || { echo "FAIL: the expf kernel did not run on the card in turbo mode: $(grep -m3 'use_expf' "$work/rel.err")"; fails=1; }
grep -q '^host_gpu_refused:' "$work/rel.err" && { echo "FAIL: turbo mode refused a kernel of this program: $(grep -m1 '^host_gpu_refused:' "$work/rel.err")"; fails=1; }
rline="$(grep '^host_gpu_turbo: mode=turbo ' "$work/rel.err" | tail -1)"
rcard="$(echo "$rline" | sed -n 's/.* launches_card=\([0-9]*\) .*/\1/p')"
(( ${rcard:-0} > ${onc:-0} )) || { echo "FAIL: turbo mode ran ${rcard:-no} launches on the card, exact $onc: the expf launches should add to them: $rline"; fails=1; }
fline="$(grep '^host_gpu_refusals: ' "$work/rel.err" | tail -1)"
echo "$fline" | grep -q 'refused_never=0 refused_atomics=0 refused_off=0 refused_exact=0 refused_safety=[1-9]' || { echo "FAIL: turbo mode's refusals: $fline"; fails=1; }
# 6. turbo verify.
run "$work/rv.out" "$work/rv.err" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=1 VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the turbo verify run failed"; tail -5 "$work/rv.err"; fails=1; }
cmp -s "$work/off.out" "$work/rv.out" || { echo "FAIL: the turbo verify run changed the program's output"; fails=1; }
vline="$(grep '^host_gpu_turbo: mode=turbo ' "$work/rv.err" | tail -1)"
vf() { echo "$vline" | sed -n "s/.* $1=\([^ ]*\).*/\1/p"; }
(( $(vf launches_card) >= 3 && $(vf words_compared) > 0 )) 2>/dev/null || { echo "FAIL: turbo verify compared nothing: $vline"; fails=1; }
[[ "$(vf words_outside)" == 0 && "$(vf kernels_outside)" == 0 ]] || { echo "FAIL: turbo verify found words outside the default tolerance: $vline"; grep '^host_gpu_outside:' "$work/rv.err" | head -5; fails=1; }
grep -Eq '^host_gpu: verify [^ ]*use_expf[^ ]* \(admitted\): [0-9]+ words compared, [0-9]+ different, 0 outside tolerance' "$work/rv.err" \
  || { echo "FAIL: the expf kernel was not verified on the card: $(grep -m2 'verify.*use_expf' "$work/rv.err")"; fails=1; }
# 7 to 9. host_gpu_vars.cu (the size rule is told to send what it can: the program's launches are short).
vrun() {   # <tag> <mode argument> [env...]
  local tag=$1 arg=$2
  shift 2
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$@" "$work/vars" "$arg" > "$work/$tag.out" 2> "$work/$tag.err"
}
send=(VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0)
for arg in vars printf tex graph; do
  vrun "voff_$arg" "$arg" || { echo "FAIL: host_gpu_vars $arg with the mode off failed"; tail -5 "$work/voff_$arg.err"; fails=1; }
done
for m in exact turbo; do
  vrun "v_$m" vars VGPU_HOST_GPU=$m "${send[@]}" || { echo "FAIL: host_gpu_vars vars, $m, failed"; tail -5 "$work/v_$m.err"; fails=1; }
  cmp -s "$work/voff_vars.out" "$work/v_$m.out" || { echo "FAIL: $m: kernels with module variables on the card change the output:"; diff "$work/voff_vars.out" "$work/v_$m.out" | head -6; fails=1; }
  for k in with_vars through_pointer; do
    grep -q "$k[^ ]* on the card" "$work/v_$m.err" || { echo "FAIL: $m: $k did not run on the card: $(grep -m2 "$k" "$work/v_$m.err")"; fails=1; }
  done
  vrun "vt_$m" tex VGPU_HOST_GPU=$m "${send[@]}" || { echo "FAIL: host_gpu_vars tex, $m, failed"; tail -5 "$work/vt_$m.err"; fails=1; }
  cmp -s "$work/voff_tex.out" "$work/vt_$m.out" || { echo "FAIL: $m: a live texture object and a launch on the card change the output"; fails=1; }
  grep -q "plain[^ ]* on the card" "$work/vt_$m.err" || { echo "FAIL: $m: a plain kernel stayed off the card with a texture object alive: $(grep -m2 'plain' "$work/vt_$m.err")"; fails=1; }
  vrun "vg_$m" graph VGPU_HOST_GPU=$m "${send[@]}" || { echo "FAIL: host_gpu_vars graph, $m, failed"; tail -5 "$work/vg_$m.err"; fails=1; }
  cmp -s "$work/voff_graph.out" "$work/vg_$m.out" || { echo "FAIL: $m: a graph's launches on the card change the output"; fails=1; }
  grep -q "plain[^ ]* on the card" "$work/vg_$m.err" || { echo "FAIL: $m: a kernel launched from a graph stayed off the card: $(grep -m2 'plain' "$work/vg_$m.err")"; fails=1; }
done
vrun vv_exact vars VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 || { echo "FAIL: host_gpu_vars vars, exact verify, failed"; fails=1; }
vl="$(grep '^host_gpu_verify:' "$work/vv_exact.err" | tail -1)"
echo "$vl" | grep -Eq 'in_set_identical=[1-9][0-9]* in_set_different=0 ' || { echo "FAIL: exact verify of the kernels with module variables: $vl"; fails=1; }
vrun vp_turbo printf VGPU_HOST_GPU=turbo "${send[@]}" || { echo "FAIL: host_gpu_vars printf, turbo, failed"; tail -5 "$work/vp_turbo.err"; fails=1; }
cmp -s "$work/voff_printf.out" "$work/vp_turbo.out" || { echo "FAIL: turbo: a kernel that prints on the card changes the output or its order:"; diff "$work/voff_printf.out" "$work/vp_turbo.out" | head -8; fails=1; }
# (nvcc 13.0 calls vprintf through a register, CALL.ABS.NOINC R2 after a MOV pair the loader patches, which the scan does not follow: the kernel
# then stays in the simulator as class never, and the step checks only that the output is unchanged; Round sp66)
if ! grep -q "prints[^ ]* on the card" "$work/vp_turbo.err"; then
  grep -q '^host_gpu_refused: kernel=[^ ]*prints[^ ]* class=never insn=CALL.ABS.NOINC R' "$work/vp_turbo.err" \
    || { echo "FAIL: turbo: the kernel that prints neither ran on the card nor was refused for a call through a register: $(grep -m2 'prints' "$work/vp_turbo.err")"; fails=1; }
fi
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/rdc" > "$work/roff.out" 2> "$work/roff.err" || { echo "FAIL: host_gpu_rdc with the mode off failed"; fails=1; }
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" VGPU_HOST_GPU=turbo "${send[@]}" "$work/rdc" > "$work/rturbo.out" 2> "$work/rturbo.err" || { echo "FAIL: host_gpu_rdc, turbo, failed"; tail -5 "$work/rturbo.err"; fails=1; }
cmp -s "$work/roff.out" "$work/rturbo.out" || { echo "FAIL: turbo: a kernel that calls into another section changes the output on the card"; fails=1; }
grep -q "calls_out[^ ]* on the card" "$work/rturbo.err" || { echo "FAIL: turbo: the kernel that calls into another section did not run on the card: $(grep -m2 'calls_out' "$work/rturbo.err")"; fails=1; }
# 12. the square root idiom on the card (Round sp68).
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets "$root/nvidia/tests/e2e/host_gpu_lazy.cu" "$root/nvidia/tests/e2e/host_gpu_lazy_heap.cu" \
  -o "$work/lazy" || { echo "FAIL: host_gpu_lazy.cu does not compile"; exit 1; }
for arg in sqrt sqrtdiv; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/lazy" "$arg" > "$work/sq_off_$arg.out" 2> /dev/null || { echo "FAIL: the $arg run with the mode off failed"; fails=1; }
  for v in 0 1; do
    env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" VGPU_HOST_GPU=exact VGPU_HOST_GPU_SQRT=1 VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_START_COST=0 \
      VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_VERIFY=$v VGPU_HOST_GPU_LOG=1 "$work/lazy" "$arg" > "$work/sq_${arg}_$v.out" 2> "$work/sq_${arg}_$v.err" \
      || { echo "FAIL: the $arg run with the square root idiom failed (verify=$v)"; tail -5 "$work/sq_${arg}_$v.err"; fails=1; }
    cmp -s "$work/sq_off_$arg.out" "$work/sq_${arg}_$v.out" || { echo "FAIL: the square root idiom changes the output of $arg on the card (verify=$v)"; fails=1; }
    grep -q 'in the exact set by the square root idiom' "$work/sq_${arg}_$v.err" || { echo "FAIL: $arg was not admitted by the square root idiom"; fails=1; }
  done
  n="$(sed -n 's/^host_gpu: \([0-9]*\) launches ran on the card.*/\1/p' "$work/sq_${arg}_0.err" | tail -1)"
  (( ${n:-0} >= 1 )) || { echo "FAIL: no launch of $arg ran on the card: $(grep '^host_gpu: [0-9]* launches' "$work/sq_${arg}_0.err")"; fails=1; }
  grep -q '^host_gpu_verify: in_set_identical=[1-9][0-9]* in_set_different=0 ' "$work/sq_${arg}_1.err" \
    || { echo "FAIL: verify of $arg: $(grep '^host_gpu_verify' "$work/sq_${arg}_1.err")"; fails=1; }
done
(( fails == 0 )) && { echo "PASS: $sum; turbo: $rline; turbo verify: $vline; module variables, printf, textures and calls into another section on the card; the square root idiom on the card"; exit 0; }
exit 1
