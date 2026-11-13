#!/usr/bin/env bash
# The host-GPU mode's start-up (include/vgpu/host_gpu.hpp should_offload_with_start; src/runtime/host_gpu.cpp await_start), tested with a stand-in driver
# (host_gpu_stub_driver.c) so that it runs on every machine, card or not. What it pins:
#   1. a program with no launch worth a card (a kernel outside the exact set, or a launch of a few warps) never loads the driver: no call reaches it;
#   2. the driver starts only when a launch has been decided to go to the card: the simulated launches before it do not wait for a driver start (the stub
#      sleeps STUB_INIT_MS in cuInit, as the real driver's context creation takes ~150 ms). VGPU_HOST_GPU_START_COST=0 (no start charged in the decision) is
#      what steps 2 to 5 use to reach the card path in a program that runs for a fraction of a second; step 7 pins the charge itself;
#   3. the card path is reached later (the stub's module load is called), and the program's output is the one it prints with the mode off;
#   4. the division idiom (include/vgpu/host_gpu.hpp division_idiom_only, on the SASS nvcc really emits for a / b): with VGPU_HOST_GPU_DIV=1 a kernel that is
#      a division reaches the card path; without it, or with a bare rcp.approx or an EX2 beside the division, it never does (not even the driver's start);
#   6. device memory outside the allocation table (a block of the kernel heap in use) keeps every launch off the card, not even the driver starts (the
#      card holds only the table, so a pointer into that memory would read what it never got);
#   5. a driver that cannot start says "mode off" once, the program's output is unchanged and no later launch calls the driver again;
#   7. the decision charges the driver start (sp43): with a start cost no launch can repay, a program whose every launch is in the exact set and big enough to
#      be a candidate never calls the driver, and its output is the mode-off output;
#   8. ... and buys it once the program has spent as much simulating as the start costs (the ski-rental account): with a start cost of half the first launch's
#      time, no single launch can repay it (the estimate is at most the first launch's time, the rule needs it above the cost plus the card path, twice over) but the
#      time already spent can, so the driver starts at the first launch that has an estimate;
#   9. VGPU_QUIET=1 silences the end-of-run summary only when nobody asked for it: with VGPU_HOST_GPU_VERIFY or VGPU_HOST_GPU_LOG set the summary lines
#      ("host_gpu: N launches ran on the card", "host_gpu_verify: in_set_identical=...") still reach stderr, and with neither set nothing is printed.
# Turbo mode (VGPU_HOST_GPU=turbo, docs/perf-design/12-host-gpu-turbo.md), with the same stand-in driver:
#  10. the kernels exact mode keeps for their numerics (expf, a bare rcp.approx, a division beside an expf) reach the card path in turbo mode, and the
#      output is unchanged (the stand-in refuses the module, so the simulator runs them);
#  11. turbo mode prints the two lines for scripts (host_gpu_turbo:, host_gpu_refusals:); exact mode prints them only with VGPU_HOST_GPU_LOG;
#  12. a float atomic with a returned value and a compare-and-swap reach the card path (Round sp66: admitted by default), and stay in the simulator (one
#      host_gpu_refused line, class=atomics) with VGPU_HOST_GPU_ATOMICS=0; an integer atomic is admitted, and VGPU_HOST_GPU_TURBO_OFF=atomics / =mufu
#      keep their classes out (class=off);
#  13. a tolerance, an off-list or a mode that cannot be read turns the mode off with a message, and nothing reaches the driver;
#  14. VGPU_HOST_GPU_VERIFY=1 in turbo mode compares by tolerance: with a stand-in card that computes nothing (STUB_CARD_OK=1) the compare reports words
#      outside tolerance, per kernel and in the summary, lists at most three launches of a kernel without VGPU_HOST_GPU_LOG, and the program's output is the
#      simulator's;
#  15. a launch that does not come back from the card (STUB_SYNC_MS) ends the process with status 124 and a message after VGPU_HOST_GPU_TIMEOUT seconds.
#  16. the simulator's launch of a verify run is not disturbed by the card copy of the launch before it: an integer reduction onto 64 contended words gives
#      the mode-off sums under VGPU_HOST_GPU_VERIFY=1 with 4 host threads, in both modes, five times over (the card copy used to stay valid through the
#      simulator's launch, whose atomics then went through the gate and lost updates).
#  17. the first blocks of a launch, timed in a forked copy of the process (Round sp60b): a launch they say is long is sent to the card; when the card
#      refuses it the simulator runs it and the output is the mode-off output (nothing the copy did is in the program's memory);
#  18. ... and when the card takes it, the card is sent the memory as it was before the launch: against a stand-in that keeps memory and computes nothing
#      (STUB_CARD_OK=1 STUB_KEEP_MEMORY=1) the program prints what the program that launches nothing prints;
#  19. a launch its first blocks say is short stays in the simulator and the driver is never touched;
#  20. a process too young to afford a sample is not sampled (the default), and VGPU_HOST_GPU_SAMPLE=log sends nothing.
#  21. progress inside the blocks (Round sp61): a launch on 8 blocks whose threads loop long is sampled inside its blocks (its line says inside=1) and
#      sent; the stand-in refuses it and the output is the mode-off output;
#  22. ... against a stand-in that keeps memory and computes nothing, the program prints what the program that launches nothing prints;
#  23. the same kernel with a short loop is sampled inside its blocks and stays, and the driver is never touched.
#  24. a launch while the card holds the live memory (Round sp65, include/vgpu/host_gpu.hpp streak_offload): against a stand-in that keeps memory and
#      computes nothing, `streak`'s short `bump` launches after the first two follow `scramble` to the card (the counters read 2, not 8; the log says why, and
#      host_gpu_streak counts them), in both modes;
#  25. ... and with VGPU_HOST_GPU_STREAK=0 they stay in the simulator (the counters read 8, nothing is counted);
#  26. when the program reads the memory back between the two (`streakread`), the card holds nothing the simulator does not and `bump` stays.
#  27. a launch on a grid of 0 blocks while the card holds the memory gets the simulator's error, as with the mode off, and never reaches the card.
#  28. (Round sp67) a verify run pushes again only the chunks whose card bytes differed from the simulator's after the last launch, and the chunks the host
#      wrote since: an allocation no kernel touches goes to the card once, a host memset of part of it goes again, and the output is the mode-off output.
#  29. (Round sp68) the square root idiom (include/vgpu/host_gpu_idiom.hpp, on the SASS nvcc emits for sqrtf): with VGPU_HOST_GPU_SQRT=1 a kernel that is a
#      square root reaches the card path, and one with a square root and a division only with VGPU_HOST_GPU_DIV=1 as well; without the switch, or with a
#      bare rsqrtf, the driver is never touched; the output is the mode-off output.
# What a module holds besides code (Round sp66; host_gpu_vars.cu, one module with __device__ and __constant__ variables, string literals, vprintf,
# malloc/free and assert):
#  30. kernels that use the module's variables (by name, and through a pointer from cudaGetSymbolAddress) reach the card path in both modes, and the
#      image the card is given has no relocation left against a variable (the mode applied them with the simulator's addresses); the output is unchanged;
#  31. ... and on a stand-in card that loads it (a verify run), the card's constant bank is looked up (cuModuleGetGlobal) and filled from the window
#      (cuMemcpyDtoD) before the first launch, and the output is the simulator's;
#  32. a kernel that prints reaches the card path in turbo mode, not in exact mode nor with VGPU_HOST_GPU_TURBO_OFF=printf; the output is unchanged;
#  33. a live texture object no longer keeps the device's launches off the card; the kernel that fetches through it stays (class never);
#  34. a kernel that calls malloc, free and assert never reaches the driver, in either mode;
#  35. a kernel that calls a device function of another section (host_gpu_rdc.cu, -rdc=true: an absolute call) reaches the card path in turbo mode,
#      whose scan follows the call into the callee's code, and not in exact mode; the output is unchanged;
#  36. a kernel launched from a CUDA graph reaches the card path in both modes; the output is unchanged.
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
# The simulator loads the card driver with RTLD_DEEPBIND, which the sanitizer runtimes refuse to dlopen.
[[ -z "$(shim_sanitizer "$shim")" ]] || { echo "SKIP: the sanitizer runtimes refuse a RTLD_DEEPBIND dlopen"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-host-gpu-stub.XXXXXX")"
trap 'rm -rf "$work"' EXIT
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$root/nvidia/tests/e2e/host_gpu_lazy.cu" "$root/nvidia/tests/e2e/host_gpu_lazy_heap.cu" -o "$work/prog" || { echo "FAIL: does not compile"; exit 1; }
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$root/nvidia/tests/e2e/host_gpu_vars.cu" -o "$work/vars" || { echo "FAIL: host_gpu_vars.cu does not compile"; exit 1; }
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -rdc=true -cudart shared -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$root/nvidia/tests/e2e/host_gpu_rdc.cu" "$root/nvidia/tests/e2e/host_gpu_rdc_fn.cu" -o "$work/rdc" || { echo "FAIL: host_gpu_rdc.cu does not compile"; exit 1; }
"$cc_bin" -shared -fPIC -O1 -o "$work/libstubcuda.so" "$root/nvidia/tests/e2e/host_gpu_stub_driver.c" || { echo "FAIL: the stand-in driver does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$work/prog"; then exit 0; fi
init_ms=2000
fails=0
run() {   # <tag> <program argument> [env...]: stdout in $work/<tag>.out, stderr in $work/<tag>.err, driver calls in $work/<tag>.log
  local tag=$1 arg=$2
  shift 2
  : > "$work/$tag.log"
  # (VGPU_HOST_GPU_SAMPLE=0: steps 1 to 16 pin the rules that decide BEFORE a launch runs, as they were before launches were sampled; steps 17 to 20
  # switch the sampling back on, their own assignment coming later on the command line.)
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" STUB_LOG="$work/$tag.log" STUB_SM="$cc" STUB_MP="$mp" STUB_INIT_MS=$init_ms \
      VGPU_HOST_GPU_LIBCUDA="$work/libstubcuda.so" VGPU_HOST_GPU_SAMPLE=0 "$@" "$work/${PROG:-prog}" "$arg" > "$work/$tag.out" 2> "$work/$tag.err"
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
# (VGPU_HOST_GPU_NO_FIRST_LAUNCH=1: the first-launch rule would send the very first launch of this looping kernel to the card, which has to wait for the driver by
# definition; what this test pins is that the simulated launches before the history rule sends one do not wait for it.)
run on_big big VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 || { echo "FAIL: the big run with the mode on failed"; tail -5 "$work/on_big.err"; fails=1; }
diff -q "$work/off_big.out" "$work/on_big.out" > /dev/null || { echo "FAIL: the mode changes the output of the big program"; fails=1; }
grep -q '^cuInit$' "$work/on_big.log" || { echo "FAIL: a launch worth a card did not start the driver: $(calls on_big)"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/on_big.log" || { echo "FAIL: no launch reached the card path (the stand-in's module load): $(calls on_big)"; fails=1; }
first="$(sed -n 's/^first_launch_ms=\([0-9]*\).*/\1/p' "$work/on_big.err" | head -1)"
[[ -n "$first" ]] || { echo "FAIL: the program printed no first_launch_ms"; fails=1; first=0; }
(( first < init_ms / 2 )) || { echo "FAIL: the first launch took $first ms: it waited for a driver start of $init_ms ms"; fails=1; }
# 4. the division idiom.
for arg in div rcp divexp; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; exit 1; }
  run "div0_$arg" "$arg" VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 || { echo "FAIL: the $arg run failed"; fails=1; }
  [[ ! -s "$work/div0_$arg.log" ]] || { echo "FAIL: $arg reached the driver without VGPU_HOST_GPU_DIV: $(calls "div0_$arg")"; fails=1; }
  run "div1_$arg" "$arg" VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 VGPU_HOST_GPU_DIV=1 || { echo "FAIL: the $arg run with the idiom on failed"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/div1_$arg.out" > /dev/null || { echo "FAIL: the idiom changes the output of $arg"; fails=1; }
  if [[ $arg == div ]]; then
    grep -q '^cuModuleLoadData$' "$work/div1_div.log" || { echo "FAIL: a plain division did not reach the card path with VGPU_HOST_GPU_DIV=1: $(calls div1_div)"; fails=1; }
  else
    [[ ! -s "$work/div1_$arg.log" ]] || { echo "FAIL: $arg (not only a division) reached the driver with VGPU_HOST_GPU_DIV=1: $(calls "div1_$arg")"; fails=1; }
  fi
done
# 5. a driver that cannot start.
run on_bad big VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 STUB_FAIL_INIT=1 || { echo "FAIL: the run with a failing driver failed"; tail -5 "$work/on_bad.err"; fails=1; }
diff -q "$work/off_big.out" "$work/on_bad.out" > /dev/null || { echo "FAIL: a driver that cannot start changes the output"; fails=1; }
n="$(grep -c 'cuInit failed; mode off' "$work/on_bad.err")"
(( n == 1 )) || { echo "FAIL: expected exactly one 'cuInit failed; mode off' line, got $n"; fails=1; }
! grep -q '^cuModuleLoadData$' "$work/on_bad.log" || { echo "FAIL: a launch reached the card after the driver failed to start"; fails=1; }
# 6. the kernel heap in use: the same program as `big` after a kernel that leaks a heap block.
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" heap > "$work/off_heap.out" 2> "$work/off_heap.err" || { echo "FAIL: the heap run with the mode off failed"; tail -5 "$work/off_heap.err"; exit 1; }
run on_heap heap VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 || { echo "FAIL: the heap run with the mode on failed"; tail -5 "$work/on_heap.err"; fails=1; }
diff -q "$work/off_heap.out" "$work/on_heap.out" > /dev/null || { echo "FAIL: the mode changes the output of the heap program"; fails=1; }
[[ ! -s "$work/on_heap.log" ]] || { echo "FAIL: a launch went to the card while the kernel heap held a block: $(calls on_heap)"; fails=1; }
# 7. the decision charges the driver start: a cost of 1000 s is never repaid, so neither the history rule nor the first-launch rule may start the driver.
for arg in big; do
  run "cost_$arg" "$arg" VGPU_HOST_GPU=exact VGPU_HOST_GPU_START_COST=1000 || { echo "FAIL: the $arg run with a start cost of 1000 s failed"; tail -5 "$work/cost_$arg.err"; fails=1; }
  [[ ! -s "$work/cost_$arg.log" ]] || { echo "FAIL: $arg reached the driver although the start could never be repaid: $(calls "cost_$arg")"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/cost_$arg.out" > /dev/null || { echo "FAIL: the start cost changes the output of $arg"; fails=1; }
done
# 8. the account: `acct` is `big` with 32 times the work per thread. A cost of half its first launch's time (read from a run whose cost is never repaid).
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" acct > "$work/off_acct.out" 2> "$work/off_acct.err" || { echo "FAIL: the acct run with the mode off failed"; exit 1; }
run cost_acct acct VGPU_HOST_GPU=exact VGPU_HOST_GPU_START_COST=1000 || { echo "FAIL: the acct run with a start cost of 1000 s failed"; fails=1; }
[[ ! -s "$work/cost_acct.log" ]] || { echo "FAIL: acct reached the driver although the start could never be repaid: $(calls cost_acct)"; fails=1; }
first_cost="$(sed -n 's/^first_launch_ms=\([0-9]*\).*/\1/p' "$work/cost_acct.err" | head -1)"
if [[ -n "$first_cost" ]] && (( first_cost >= 8 )); then
  half="$(awk -v ms="$first_cost" 'BEGIN { printf "%.6f", ms / 2000 }')"
  run acct_run acct VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST="$half" || { echo "FAIL: the account run failed"; tail -5 "$work/acct_run.err"; fails=1; }
  grep -q '^cuInit$' "$work/acct_run.log" || { echo "FAIL: a program that had spent more than the start cost ($half s) in the simulator did not start the driver: $(calls acct_run)"; fails=1; }
  diff -q "$work/off_acct.out" "$work/acct_run.out" > /dev/null || { echo "FAIL: the account changes the output of acct"; fails=1; }
else
  echo "FAIL: the first launch of acct took ${first_cost:-?} ms: too short to halve (the program does 32 times the work of big)"; fails=1
fi
# 9. the summary under VGPU_QUIET=1.
run q_none big VGPU_QUIET=1 VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 || { echo "FAIL: the quiet run failed"; fails=1; }
grep -q 'launches ran on the card\|host_gpu_verify:' "$work/q_none.err" && { echo "FAIL: VGPU_QUIET=1 alone printed the summary"; fails=1; }
run q_log big VGPU_QUIET=1 VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 || { echo "FAIL: the quiet run with VGPU_HOST_GPU_LOG failed"; fails=1; }
grep -Eq '^host_gpu: [0-9]+ launches ran on the card' "$work/q_log.err" || { echo "FAIL: VGPU_QUIET=1 with VGPU_HOST_GPU_LOG=1 printed no 'launches ran on the card' line"; fails=1; }
run q_ver big VGPU_QUIET=1 VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 || { echo "FAIL: the quiet verify run failed"; fails=1; }
grep -Eq '^host_gpu_verify: in_set_identical=[0-9]+ in_set_different=[0-9]+ other_identical=[0-9]+ other_different=[0-9]+$' "$work/q_ver.err" || { echo "FAIL: VGPU_QUIET=1 with VGPU_HOST_GPU_VERIFY=1 printed no host_gpu_verify summary line"; fails=1; }
grep -Eq '^host_gpu: [0-9]+ launches ran on the card' "$work/q_ver.err" || { echo "FAIL: VGPU_QUIET=1 with VGPU_HOST_GPU_VERIFY=1 printed no 'launches ran on the card' line"; fails=1; }
diff -q "$work/off_big.out" "$work/q_ver.out" > /dev/null || { echo "FAIL: verify under VGPU_QUIET changes the output"; fails=1; }
# ---- turbo mode ----
reach=(VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0)
for arg in iatom fatom cas; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; tail -5 "$work/off_$arg.err"; exit 1; }
done
# 10. what exact keeps for its numerics goes to the card in turbo.
for arg in mufu rcp divexp; do
  run "rel_$arg" "$arg" VGPU_HOST_GPU=turbo "${reach[@]}" || { echo "FAIL: the turbo $arg run failed"; tail -5 "$work/rel_$arg.err"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/rel_$arg.log" || { echo "FAIL: $arg did not reach the card path in turbo mode: $(calls "rel_$arg"); $(grep -m2 'host_gpu' "$work/rel_$arg.err")"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/rel_$arg.out" > /dev/null || { echo "FAIL: turbo mode with a card that refuses the module changes the output of $arg"; fails=1; }
  grep -q '^host_gpu_refused:' "$work/rel_$arg.err" && { echo "FAIL: turbo mode refused a kernel of $arg: $(grep -m1 '^host_gpu_refused:' "$work/rel_$arg.err")"; fails=1; }
done
run ex_mufu mufu VGPU_HOST_GPU=exact "${reach[@]}" || { echo "FAIL: the exact mufu run failed"; fails=1; }
[[ ! -s "$work/ex_mufu.log" ]] || { echo "FAIL: exact mode sent the expf kernel to the card: $(calls ex_mufu)"; fails=1; }
# 11. the lines for scripts.
rl='^host_gpu_turbo: mode=turbo launches_card=[0-9]+ launches_sim=[0-9]+ card_s=[0-9.]+ sim_s=[0-9.]+ words_compared=[0-9]+ words_different=[0-9]+ words_outside=[0-9]+ words_subnormal=[0-9]+ kernels_outside=[0-9]+ max_abs=[^ ]+ max_rel=[^ ]+$'
fl='^host_gpu_refusals: refused_never=[0-9]+ refused_atomics=[0-9]+ refused_off=[0-9]+ refused_exact=[0-9]+ refused_safety=[0-9]+ too_small=[0-9]+ failed=[0-9]+$'
grep -Eq "$rl" "$work/rel_mufu.err" || { echo "FAIL: turbo mode printed no host_gpu_turbo line: $(grep '^host_gpu_turbo' "$work/rel_mufu.err")"; fails=1; }
grep -Eq "$fl" "$work/rel_mufu.err" || { echo "FAIL: turbo mode printed no host_gpu_refusals line: $(grep '^host_gpu_refusals' "$work/rel_mufu.err")"; fails=1; }
grep -Eq '^host_gpu: [0-9]+ launches ran on the card, [0-9]+ in the simulator \([0-9]+ kernels not admitted by turbo mode' "$work/rel_mufu.err" || { echo "FAIL: turbo mode's summary line is missing"; fails=1; }
grep -q '^host_gpu_turbo:\|^host_gpu_refusals:\|^host_gpu_refused:' "$work/on_big.err" "$work/ex_mufu.err" && { echo "FAIL: exact mode without VGPU_HOST_GPU_LOG printed turbo mode's lines"; fails=1; }
grep -Eq '^host_gpu: [0-9]+ launches ran on the card, [0-9]+ in the simulator \([0-9]+ kernels not in the exact set, [0-9]+ too small, [0-9]+ unsupported launch shapes, [0-9]+ failed on the card\)$' "$work/ex_mufu.err" || { echo "FAIL: exact mode's summary line changed: $(grep 'launches ran' "$work/ex_mufu.err")"; fails=1; }
grep -Eq '^host_gpu_turbo: mode=exact ' "$work/q_log.err" || { echo "FAIL: exact mode with VGPU_HOST_GPU_LOG=1 printed no host_gpu_turbo line"; fails=1; }
run exl_mufu mufu VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 "${reach[@]}" || { echo "FAIL: the exact mufu run with the log failed"; fails=1; }
rx="$(sed -n 's/^host_gpu_refusals: .* refused_exact=\([0-9]*\) .*/\1/p' "$work/exl_mufu.err" | tail -1)"
(( ${rx:-0} >= 1 )) || { echo "FAIL: exact mode did not count the expf launches as ones turbo mode would take: $(grep '^host_gpu_refusals' "$work/exl_mufu.err")"; fails=1; }
# 12. atomics: on by default (Round sp66), VGPU_HOST_GPU_ATOMICS=0 keeps them out, and the off list.
for arg in fatom cas; do
  run "rel_$arg" "$arg" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_ATOMICS=0 "${reach[@]}" || { echo "FAIL: the turbo $arg run failed"; tail -5 "$work/rel_$arg.err"; fails=1; }
  [[ ! -s "$work/rel_$arg.log" ]] || { echo "FAIL: $arg reached the driver with VGPU_HOST_GPU_ATOMICS=0: $(calls "rel_$arg")"; fails=1; }
  n="$(grep -c '^host_gpu_refused: kernel=[^ ]* class=atomics insn=.' "$work/rel_$arg.err")"
  (( n == 1 )) || { echo "FAIL: $arg: expected one host_gpu_refused line with class=atomics, got $n: $(grep -m2 '^host_gpu_refused' "$work/rel_$arg.err")"; fails=1; }
  ra="$(sed -n 's/^host_gpu_refusals: .* refused_atomics=\([0-9]*\) .*/\1/p' "$work/rel_$arg.err" | tail -1)"
  (( ${ra:-0} == 4 )) || { echo "FAIL: $arg: refused_atomics is ${ra:-missing}, not the 4 launches"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/rel_$arg.out" > /dev/null || { echo "FAIL: turbo mode changes the output of $arg"; fails=1; }
  run "rela_$arg" "$arg" VGPU_HOST_GPU=turbo "${reach[@]}" || { echo "FAIL: the turbo $arg run with the default atomics failed"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/rela_$arg.log" || { echo "FAIL: $arg did not reach the card path by default: $(calls "rela_$arg")"; fails=1; }
  grep -q '^host_gpu_refused:' "$work/rela_$arg.err" && { echo "FAIL: $arg was refused by default: $(grep -m1 '^host_gpu_refused:' "$work/rela_$arg.err")"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/rela_$arg.out" > /dev/null || { echo "FAIL: the default atomics change the output of $arg"; fails=1; }
done
run rel_iatom iatom VGPU_HOST_GPU=turbo "${reach[@]}" || { echo "FAIL: the turbo iatom run failed"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/rel_iatom.log" || { echo "FAIL: an integer atomic did not reach the card path in turbo mode: $(calls rel_iatom); $(grep -m1 '^host_gpu_refused' "$work/rel_iatom.err")"; fails=1; }
diff -q "$work/off_iatom.out" "$work/rel_iatom.out" > /dev/null || { echo "FAIL: turbo mode changes the output of iatom"; fails=1; }
for pair in iatom:atomics mufu:mufu; do
  arg="${pair%%:*}"; cls="${pair##*:}"
  run "cls_$cls" "$arg" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TURBO_OFF="call,$cls" "${reach[@]}" || { echo "FAIL: the run with VGPU_HOST_GPU_TURBO_OFF=$cls failed"; fails=1; }
  [[ ! -s "$work/cls_$cls.log" ]] || { echo "FAIL: $arg reached the driver with its class switched off: $(calls "cls_$cls")"; fails=1; }
  grep -q '^host_gpu_refused: kernel=[^ ]* class=off insn=.' "$work/cls_$cls.err" || { echo "FAIL: no host_gpu_refused line with class=off for $arg"; fails=1; }
  diff -q "$work/off_$arg.out" "$work/cls_$cls.out" > /dev/null || { echo "FAIL: VGPU_HOST_GPU_TURBO_OFF=$cls changes the output of $arg"; fails=1; }
done
# 13. values that cannot be read: the mode is off, said once, and the driver is never touched.
bad() {   # <tag> <the message's text> [env...]
  local tag=$1 text=$2
  shift 2
  run "$tag" mufu VGPU_QUIET=1 "${reach[@]}" "$@" || { echo "FAIL: the $tag run failed"; fails=1; }
  [[ ! -s "$work/$tag.log" ]] || { echo "FAIL: $tag: the driver was reached: $(calls "$tag")"; fails=1; }
  local n
  n="$(grep -c "$text" "$work/$tag.err")"
  (( n == 1 )) || { echo "FAIL: $tag: expected one line with '$text', got $n: $(grep -m2 host_gpu "$work/$tag.err")"; fails=1; }
  grep -q 'mode off$' "$work/$tag.err" || { echo "FAIL: $tag: no 'mode off'"; fails=1; }
  diff -q "$work/off_mufu.out" "$work/$tag.out" > /dev/null || { echo "FAIL: $tag changes the output"; fails=1; }
}
bad bad_atol 'VGPU_HOST_GPU_ATOL=-1e-6 is not a number' VGPU_HOST_GPU=turbo VGPU_HOST_GPU_ATOL=-1e-6
bad bad_rtol 'VGPU_HOST_GPU_RTOL=1e-5x is not a number' VGPU_HOST_GPU=turbo VGPU_HOST_GPU_RTOL=1e-5x
bad bad_off 'VGPU_HOST_GPU_TURBO_OFF: sfu is not a class' VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TURBO_OFF=mufu,sfu
bad bad_time 'VGPU_HOST_GPU_TIMEOUT=soon is not a number' VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TIMEOUT=soon
bad bad_type 'VGPU_HOST_GPU_TOL_TYPE=half is not auto, f32, f64 or f16' VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TOL_TYPE=half
bad bad_mode 'VGPU_HOST_GPU=approx is not a mode (off, exact or turbo)' VGPU_HOST_GPU=approx
# (exact mode does not read turbo mode's tolerances: a bad one there is nobody's)
run ex_atol big VGPU_HOST_GPU=exact VGPU_HOST_GPU_ATOL=-1 "${reach[@]}" || { echo "FAIL: the exact run with a turbo-only variable failed"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/ex_atol.log" || { echo "FAIL: exact mode was switched off by VGPU_HOST_GPU_ATOL"; fails=1; }
# 14. verify by tolerance, against a card that computes nothing.
run rel_ver mufu VGPU_QUIET=1 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 "${reach[@]}" || { echo "FAIL: the turbo verify run failed"; tail -5 "$work/rel_ver.err"; fails=1; }
diff -q "$work/off_mufu.out" "$work/rel_ver.out" > /dev/null || { echo "FAIL: turbo verify changes the output (the simulator's result is the program's)"; fails=1; }
line="$(grep '^host_gpu_turbo:' "$work/rel_ver.err" | tail -1)"
field() { echo "$line" | sed -n "s/.* $1=\([^ ]*\).*/\1/p"; }
wc_="$(field words_compared)"; wd="$(field words_different)"; wo="$(field words_outside)"; ko="$(field kernels_outside)"; lc="$(field launches_card)"
(( ${lc:-0} == 4 && ${wc_:-0} >= 4 * 262144 )) || { echo "FAIL: turbo verify compared ${wc_:-no} words over ${lc:-no} card launches: $line"; fails=1; }
(( ${wo:-0} > 0 && ${wd:-0} >= ${wo:-0} && ${ko:-0} == 1 )) || { echo "FAIL: a card that computed nothing was not reported outside tolerance: $line"; fails=1; }
n="$(grep -c '^host_gpu: verify .* outside tolerance' "$work/rel_ver.err")"
(( n == 3 )) || { echo "FAIL: expected the first three launches outside tolerance to be listed without the log, got $n"; fails=1; }
grep -Eq '^host_gpu: verify .*: [0-9]+ words compared, [0-9]+ different, [1-9][0-9]* outside tolerance, .* first outside at 0x[0-9a-f]+: card 0x[0-9a-f]{8} sim 0x[0-9a-f]{8}' "$work/rel_ver.err" || { echo "FAIL: the per-launch report has no first address: $(grep -m1 '^host_gpu: verify' "$work/rel_ver.err")"; fails=1; }
grep -Eq '^host_gpu_outside: kernel=[^ ]+ words_compared=[0-9]+ words_different=[0-9]+ words_outside=[1-9][0-9]* max_abs=[^ ]+ max_rel=[^ ]+ first=0x[0-9a-f]+:card=0x[0-9a-f]{8}:sim=0x[0-9a-f]{8}$' "$work/rel_ver.err" || { echo "FAIL: no host_gpu_outside line"; fails=1; }
grep -Eq '^host_gpu_verify: in_set_identical=0 in_set_different=4 other_identical=0 other_different=0$' "$work/rel_ver.err" || { echo "FAIL: turbo verify's host_gpu_verify line: $(grep '^host_gpu_verify' "$work/rel_ver.err")"; fails=1; }
# ... and exact mode's verify of a kernel of its set is what it was: the stand-in's empty card differs, in 64 KiB chunks, with no tolerance report.
run ex_ver big VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 "${reach[@]}" || { echo "FAIL: the exact verify run failed"; fails=1; }
grep -Eq '^host_gpu: verify [^ ]+ \(in the exact set\): DIFFERENT, 64 KiB chunks: [0-9]+$' "$work/ex_ver.err" || { echo "FAIL: exact verify's per-launch line changed: $(grep -m1 '^host_gpu: verify' "$work/ex_ver.err")"; fails=1; }
grep -q 'outside tolerance\|^host_gpu_turbo:\|^host_gpu_outside:' "$work/ex_ver.err" && { echo "FAIL: exact verify printed turbo mode's report"; fails=1; }
# 15. a kernel that does not come back.
t0=$SECONDS
run hang mufu VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TIMEOUT=1 STUB_CARD_OK=1 STUB_SYNC_MS=20000 "${reach[@]}"
rc=$?
(( rc == 124 )) || { echo "FAIL: a launch that hung on the card ended with status $rc, not 124"; tail -3 "$work/hang.err"; fails=1; }
(( SECONDS - t0 < 15 )) || { echo "FAIL: the hung launch was waited for ($(( SECONDS - t0 )) s)"; fails=1; }
grep -q 'has been on the card for more than 1 s (VGPU_HOST_GPU_TIMEOUT)' "$work/hang.err" || { echo "FAIL: no message for the hung launch"; fails=1; }
# (a verify run, so that the launches do reach the stand-in's wait and the output is still the simulator's)
run nohang mufu VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=1 VGPU_HOST_GPU_TIMEOUT=30 STUB_CARD_OK=1 STUB_SYNC_MS=50 "${reach[@]}" || { echo "FAIL: a launch inside its time limit ended the process"; tail -3 "$work/nohang.err"; fails=1; }
grep -q '^cuLaunchKernel$' "$work/nohang.log" || { echo "FAIL: the run inside the time limit launched nothing on the stand-in card"; fails=1; }
diff -q "$work/off_mufu.out" "$work/nohang.out" > /dev/null || { echo "FAIL: the time limit changes the output"; fails=1; }
# 16. verify with contended atomics in the simulator.
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" hist > "$work/off_hist.out" 2> "$work/off_hist.err" || { echo "FAIL: the hist run with the mode off failed"; exit 1; }
for m in exact turbo; do
  for i in 1 2 3 4 5; do
    run "hist_$m" hist VGPU_HOST_GPU=$m VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 "${reach[@]}" || { echo "FAIL: the $m verify run of hist failed"; fails=1; }
    diff -q "$work/off_hist.out" "$work/hist_$m.out" > /dev/null || { echo "FAIL: $m verify changes the sums of an integer reduction (run $i): the simulator lost updates"; fails=1; break; }
  done
  grep -q '^cuLaunchKernel$' "$work/hist_$m.log" || { echo "FAIL: the $m verify run of hist launched nothing on the stand-in card"; fails=1; }
done
# 17 to 20. the first blocks of a launch, timed (include/vgpu/host_gpu.hpp "The first blocks of a launch, timed"). VGPU_HOST_GPU_SAMPLE=always: this
# program launches within milliseconds of its start, and by default a process that young is not delayed by a sample (step 20).
sampled_of() { sed -n 's/^host_gpu_sampled: sampled=\([0-9]*\) sent=\([0-9]*\) .*/\1 \2/p' "$work/$1.err" | head -1; }
smp=(VGPU_HOST_GPU=exact VGPU_HOST_GPU_SAMPLE=always VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 STUB_INIT_MS=50)
# 17. a launch that its first blocks say is long is sent; the stand-in refuses the module, the simulator runs the launch, and the output is the mode-off
# output (the kernel reads what it writes: a block the copy ran that showed in the program's memory would change the hashes).
run smp_acct acct "${smp[@]}" VGPU_HOST_GPU_START_COST=0 || { echo "FAIL: the sampled run of acct failed"; tail -5 "$work/smp_acct.err"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of smp_acct)"
(( ${smp_n:-0} >= 1 )) || { echo "FAIL: no launch of acct was sampled: $(grep host_gpu_sampled "$work/smp_acct.err")"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/smp_acct.log" || { echo "FAIL: the sampled launch of acct did not reach the card path: $(calls smp_acct)"; fails=1; }
diff -q "$work/off_acct.out" "$work/smp_acct.out" > /dev/null || { echo "FAIL: a sampled launch that the card refused changes the output of acct"; fails=1; }
# 18. the same against a card that keeps memory and computes nothing: the launch is sent, and the program reads the memory as it was before the launch:
# the output of the program that launches nothing.
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" none > "$work/off_none.out" 2> "$work/off_none.err" || { echo "FAIL: the none run with the mode off failed"; exit 1; }
run smp_nop acct "${smp[@]}" VGPU_HOST_GPU_START_COST=0 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the sampled run of acct on a card that computes nothing failed"; tail -5 "$work/smp_nop.err"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of smp_nop)"
(( ${smp_sent:-0} >= 1 )) || { echo "FAIL: the sampled launch of acct was not sent: $(grep host_gpu_sampled "$work/smp_nop.err")"; fails=1; }
diff -q "$work/off_none.out" "$work/smp_nop.out" > /dev/null || { echo "FAIL: a sampled launch sent to a card that computes nothing does not leave the memory as it was before the launch"; fails=1; }
diff -q "$work/off_acct.out" "$work/smp_nop.out" > /dev/null && { echo "FAIL: the run on a card that computes nothing printed the simulator's results: nothing was sent"; fails=1; }
# 19. a launch that its first blocks say is short stays, and the driver is never touched (the default start cost: nothing here repays 0.25 s).
run smp_big big "${smp[@]}" || { echo "FAIL: the sampled run of big failed"; tail -5 "$work/smp_big.err"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of smp_big)"
(( ${smp_n:-0} >= 1 && ${smp_sent:-1} == 0 )) || { echo "FAIL: big: expected a sampled launch and none sent: $(grep host_gpu_sampled "$work/smp_big.err")"; fails=1; }
[[ ! -s "$work/smp_big.log" ]] || { echo "FAIL: a short launch reached the driver after its sample: $(calls smp_big)"; fails=1; }
diff -q "$work/off_big.out" "$work/smp_big.out" > /dev/null || { echo "FAIL: sampling changes the output of big"; fails=1; }
# 20. by default a process this young is not sampled at all; and `log` samples but sends nothing, even a launch worth sending.
run smp_young acct VGPU_HOST_GPU=exact VGPU_HOST_GPU_SAMPLE=1 VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=1000 || { echo "FAIL: the default run of acct failed"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of smp_young)"
[[ "${smp_n:-x}" == 0 ]] || { echo "FAIL: a process a few milliseconds old was sampled: $(grep host_gpu_sampled "$work/smp_young.err")"; fails=1; }
run smp_log acct VGPU_HOST_GPU=exact VGPU_HOST_GPU_SAMPLE=log VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=1000 || { echo "FAIL: the log run of acct failed"; fails=1; }
grep -q '^host_gpu_sample: ' "$work/smp_log.err" || { echo "FAIL: VGPU_HOST_GPU_SAMPLE=log printed no host_gpu_sample line"; fails=1; }
[[ ! -s "$work/smp_log.log" ]] || { echo "FAIL: a log run reached the driver: $(calls smp_log)"; fails=1; }
diff -q "$work/off_acct.out" "$work/smp_log.out" > /dev/null || { echo "FAIL: the log run changes the output of acct"; fails=1; }
# 21 to 23. progress inside the blocks (include/vgpu/host_gpu.hpp "progress inside a block"): `few` is 8 blocks of 256 threads, each looping 32,768 times.
inside_of() { grep '^host_gpu_sample: ' "$work/$1.err" | grep -c ' inside=1 '; }
for arg in few fewshort fewnone; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; exit 1; }
done
run in_few few "${smp[@]}" VGPU_HOST_GPU_START_COST=0 || { echo "FAIL: the sampled run of few failed"; tail -5 "$work/in_few.err"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of in_few)"
(( ${smp_n:-0} >= 1 && $(inside_of in_few) >= 1 )) || { echo "FAIL: no launch of few was sampled inside its blocks: $(grep -E 'host_gpu_sampled|host_gpu_sample:' "$work/in_few.err" | head -3)"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/in_few.log" || { echo "FAIL: the launch of few sampled inside its blocks did not reach the card path: $(calls in_few)"; fails=1; }
diff -q "$work/off_few.out" "$work/in_few.out" > /dev/null || { echo "FAIL: a launch sampled inside its blocks that the card refused changes the output of few"; fails=1; }
run in_nop few "${smp[@]}" VGPU_HOST_GPU_START_COST=0 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the run of few on a card that computes nothing failed"; tail -5 "$work/in_nop.err"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of in_nop)"
(( ${smp_sent:-0} >= 1 )) || { echo "FAIL: the launch of few sampled inside its blocks was not sent: $(grep host_gpu_sampled "$work/in_nop.err")"; fails=1; }
diff -q "$work/off_fewnone.out" "$work/in_nop.out" > /dev/null || { echo "FAIL: a launch sampled inside its blocks and sent to a card that computes nothing does not leave the memory as it was"; fails=1; }
run in_short fewshort "${smp[@]}" || { echo "FAIL: the sampled run of fewshort failed"; tail -5 "$work/in_short.err"; fails=1; }
read -r smp_n smp_sent <<< "$(sampled_of in_short)"
(( ${smp_n:-0} >= 1 && ${smp_sent:-1} == 0 && $(inside_of in_short) >= 1 )) || { echo "FAIL: fewshort: expected a launch sampled inside its blocks and none sent: $(grep -E 'host_gpu_sampled|host_gpu_sample:' "$work/in_short.err" | head -3)"; fails=1; }
[[ ! -s "$work/in_short.log" ]] || { echo "FAIL: a short launch sampled inside its blocks reached the driver: $(calls in_short)"; fails=1; }
diff -q "$work/off_fewshort.out" "$work/in_short.out" > /dev/null || { echo "FAIL: sampling inside the blocks changes the output of fewshort"; fails=1; }
# 24 to 26. a launch while the card holds the live memory. The counters are what a card that computes nothing leaves: 2 when every bump after the
# first two (the history) went to the card, 8 when none did.
streak_of() { sed -n 's/^host_gpu_streak: sent=\([0-9]*\) card_only_mib=[0-9.]* rule=\([01]\)$/\1 \2/p' "$work/$1.err" | head -1; }
str=(VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 STUB_INIT_MS=50)
for arg in streak streakread; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; exit 1; }
  grep -q '^bumps 8 to 8, ' "$work/off_$arg.out" || { echo "FAIL: $arg with the mode off: $(cat "$work/off_$arg.out")"; fails=1; }
done
for m in exact turbo; do
  run "str_$m" streak VGPU_HOST_GPU=$m "${str[@]}" || { echo "FAIL: the $m run of streak failed"; tail -5 "$work/str_$m.err"; fails=1; }
  grep -q '^bumps 2 to 2, ' "$work/str_$m.out" || { echo "FAIL: $m: the bumps after the first two did not follow scramble to the card: $(cat "$work/str_$m.out"); $(grep -m3 'bump' "$work/str_$m.err")"; fails=1; }
  read -r st_n st_rule <<< "$(streak_of "str_$m")"
  [[ "${st_n:-x} ${st_rule:-x}" == "6 1" ]] || { echo "FAIL: $m: host_gpu_streak says ${st_n:-nothing} sent (rule ${st_rule:-?}), not 6: $(grep '^host_gpu_streak' "$work/str_$m.err")"; fails=1; }
  n="$(grep -c 'bump.*: sent because the card holds the memory' "$work/str_$m.err")"
  (( n == 6 )) || { echo "FAIL: $m: expected 6 lines saying why bump was sent, got $n"; fails=1; }
done
run str_off streak VGPU_HOST_GPU=exact VGPU_HOST_GPU_STREAK=0 "${str[@]}" || { echo "FAIL: the run of streak with VGPU_HOST_GPU_STREAK=0 failed"; fails=1; }
grep -q '^bumps 8 to 8, ' "$work/str_off.out" || { echo "FAIL: with VGPU_HOST_GPU_STREAK=0 a bump went to the card: $(cat "$work/str_off.out")"; fails=1; }
[[ "$(streak_of str_off)" == "0 0" ]] || { echo "FAIL: with VGPU_HOST_GPU_STREAK=0: $(grep '^host_gpu_streak' "$work/str_off.err")"; fails=1; }
diff -q "$work/off_streak.out" "$work/str_off.out" > /dev/null && { echo "FAIL: scramble did not go to the card in the streak runs (the hash is the simulator's)"; fails=1; }
run str_read streakread VGPU_HOST_GPU=exact "${str[@]}" || { echo "FAIL: the run of streakread failed"; fails=1; }
grep -q '^bumps 8 to 8, ' "$work/str_read.out" || { echo "FAIL: a bump went to the card although the program had read the memory back: $(cat "$work/str_read.out")"; fails=1; }
[[ "$(streak_of str_read)" == "0 1" ]] || { echo "FAIL: streakread: $(grep '^host_gpu_streak' "$work/str_read.err")"; fails=1; }
# 27. a zero grid while the card holds the memory (Round sp65: PolyBench LU's last lu_kernel1).
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" zerogrid > "$work/off_zerogrid.out" 2> "$work/off_zerogrid.err" || { echo "FAIL: the zerogrid run with the mode off failed"; exit 1; }
grep -q '^zero grid: cudaErrorInvalidConfiguration$' "$work/off_zerogrid.out" || { echo "FAIL: zerogrid with the mode off: $(cat "$work/off_zerogrid.out")"; fails=1; }
for m in exact turbo; do
  run "zg_$m" zerogrid VGPU_HOST_GPU=$m "${str[@]}" || { echo "FAIL: the $m run of zerogrid failed"; tail -5 "$work/zg_$m.err"; fails=1; }
  diff -q "$work/off_zerogrid.out" "$work/zg_$m.out" > /dev/null || { echo "FAIL: $m: a zero grid did not get the simulator's answer: $(cat "$work/zg_$m.out")"; fails=1; }
  grep -q 'a grid or block dimension is 0' "$work/zg_$m.err" || { echo "FAIL: $m: no line saying the zero grid stayed in the simulator"; fails=1; }
  grep -q '^cuLaunchKernel$' "$work/zg_$m.log" || { echo "FAIL: $m: nothing went to the card before the zero grid, so the step proves nothing: $(calls "zg_$m")"; fails=1; }
done
# 28. (Round sp67; 24 to 27 are Round sp65's) a verify run gives the card back the chunks whose bytes agreed after a launch, and pushes only the others
# before the next one. `keep` is `big` (1 MiB, every chunk written by every launch) beside 8 MiB no kernel touches, with a host memset of the first 64 KiB of
# those after the second launch. Against a card that keeps memory and computes nothing, every chunk of the 1 MiB differs after every launch (16 chunks) and every
# chunk of the 8 MiB agrees: the 9 MiB go once, then the 1 MiB before each launch, and the memset chunk once more: 12.1 MiB in all (it was 36.0: every byte
# before every launch). The memset chunk must reach the card before the third launch (else the compare finds a 17th chunk different), and the output is the
# mode-off output.
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" keep > "$work/off_keep.out" 2> "$work/off_keep.err" || { echo "FAIL: the keep run with the mode off failed"; exit 1; }
for m in exact turbo; do
  run "keep_$m" keep VGPU_HOST_GPU=$m VGPU_HOST_GPU_VERIFY=1 VGPU_HOST_GPU_LOG=1 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 "${reach[@]}" || { echo "FAIL: the $m verify run of keep failed"; tail -5 "$work/keep_$m.err"; fails=1; }
  diff -q "$work/off_keep.out" "$work/keep_$m.out" > /dev/null || { echo "FAIL: $m verify changes the output of keep"; fails=1; }
  pushed="$(sed -n 's/^host_gpu: \([0-9.]*\) MiB went to the card.*/\1/p' "$work/keep_$m.err" | tail -1)"
  [[ "$pushed" == 12.1 ]] || { echo "FAIL: $m verify of keep pushed ${pushed:-?} MiB, not 12.1 (9 once, 1 per later launch, the memset chunk once)"; fails=1; }
  grep -q '^host_gpu_verify: in_set_identical=0 in_set_different=4 ' "$work/keep_$m.err" || { echo "FAIL: $m verify of keep: $(grep '^host_gpu_verify' "$work/keep_$m.err")"; fails=1; }
done
# (exact mode names the chunks: the 16 of the 1 MiB after every launch, never a 17th: the memset chunk reached the card before the third launch)
n="$(grep -c '^host_gpu: verify [^ ]* (in the exact set): DIFFERENT, 64 KiB chunks: 16$' "$work/keep_exact.err")"
(( n == 4 )) || { echo "FAIL: exact verify of keep: expected 4 launches with the 16 chunks of the 1 MiB different, got $n: $(grep '^host_gpu: verify' "$work/keep_exact.err" | head -4)"; fails=1; }
# 29. (Round sp68) the square root idiom: `sqrt` (sqrtf on both of its paths), `rsqrt` (a bare rsqrtf), `sqrtdiv` (sqrtf, then a division).
for arg in sqrt rsqrt sqrtdiv; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; exit 1; }
  run "sq0_$arg" "$arg" VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 VGPU_HOST_GPU_DIV=1 || { echo "FAIL: the $arg run failed"; fails=1; }
  [[ ! -s "$work/sq0_$arg.log" ]] || { echo "FAIL: $arg reached the driver without VGPU_HOST_GPU_SQRT: $(calls "sq0_$arg")"; fails=1; }
  for both in 0 1; do
    run "sq1_${arg}_$both" "$arg" VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 VGPU_HOST_GPU_SQRT=1 VGPU_HOST_GPU_DIV=$both \
      VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the $arg run with the square root idiom on failed"; fails=1; }
    diff -q "$work/off_$arg.out" "$work/sq1_${arg}_$both.out" > /dev/null || { echo "FAIL: the square root idiom changes the output of $arg"; fails=1; }
    want=no
    [[ $arg == sqrt || ( $arg == sqrtdiv && $both == 1 ) ]] && want=yes
    if [[ $want == yes ]]; then
      grep -q '^cuModuleLoadData$' "$work/sq1_${arg}_$both.log" || { echo "FAIL: $arg did not reach the card path with VGPU_HOST_GPU_SQRT=1 DIV=$both: $(calls "sq1_${arg}_$both")"; fails=1; }
      grep -q 'in the exact set by the square root idiom (1 site)' "$work/sq1_${arg}_$both.err" || { echo "FAIL: $arg: no log line for the idiom: $(grep 'host_gpu: ' "$work/sq1_${arg}_$both.err" | head -3)"; fails=1; }
    else
      [[ ! -s "$work/sq1_${arg}_$both.log" ]] || { echo "FAIL: $arg reached the driver with VGPU_HOST_GPU_SQRT=1 DIV=$both: $(calls "sq1_${arg}_$both")"; fails=1; }
    fi
  done
done
# 30 to 34. what a module holds besides code (host_gpu_vars.cu).
for arg in vars printf tex heap graph; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/vars" "$arg" > "$work/off_v$arg.out" 2> "$work/off_v$arg.err" || { echo "FAIL: the $arg run of host_gpu_vars with the mode off failed"; tail -5 "$work/off_v$arg.err"; exit 1; }
done
grep -q '^vars 5: ' "$work/off_vvars.out" || { echo "FAIL: host_gpu_vars vars printed: $(head -2 "$work/off_vvars.out")"; fails=1; }
for m in exact turbo; do
  PROG=vars run "v_$m" vars VGPU_HOST_GPU=$m VGPU_HOST_GPU_LOG=1 "${reach[@]}" || { echo "FAIL: the $m vars run failed"; tail -5 "$work/v_$m.err"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/v_$m.log" || { echo "FAIL: $m: kernels using module variables did not reach the card path: $(calls "v_$m"); $(grep -m2 'stays in the simulator' "$work/v_$m.err")"; fails=1; }
  grep -q '^relocs_to_variables 0$' "$work/v_$m.log" || { echo "FAIL: $m: the card's image still has relocations against the module's variables: $(grep relocs_to "$work/v_$m.log" | head -1)"; fails=1; }
  diff -q "$work/off_vvars.out" "$work/v_$m.out" > /dev/null || { echo "FAIL: $m changes the output of host_gpu_vars vars"; fails=1; }
  PROG=vars run "vv_$m" vars VGPU_HOST_GPU=$m VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 "${reach[@]}" || { echo "FAIL: the $m vars verify run failed"; tail -5 "$work/vv_$m.err"; fails=1; }
  order="$(grep -E '^(cuModuleGetGlobal|cuMemcpyDtoD|cuLaunchKernel)$' "$work/vv_$m.log" | uniq | head -3 | tr '\n' ' ')"
  [[ "$order" == "cuModuleGetGlobal cuMemcpyDtoD cuLaunchKernel " ]] || { echo "FAIL: $m: the constant bank was not looked up and filled before the first launch on the card: $order"; fails=1; }
  diff -q "$work/off_vvars.out" "$work/vv_$m.out" > /dev/null || { echo "FAIL: $m verify changes the output of host_gpu_vars vars"; fails=1; }
done
PROG=vars run vp_turbo printf VGPU_HOST_GPU=turbo "${reach[@]}" || { echo "FAIL: the turbo printf run failed"; tail -5 "$work/vp_turbo.err"; fails=1; }
# (nvcc 13.0 calls vprintf through a register, which the scan does not follow: then the kernel is refused as class never; Round sp66)
grep -q '^cuModuleLoadData$' "$work/vp_turbo.log" || grep -q '^host_gpu_refused: kernel=[^ ]*prints[^ ]* class=never insn=CALL.ABS.NOINC R' "$work/vp_turbo.err" \
  || { echo "FAIL: a kernel that prints neither reached the card path in turbo mode nor was refused for a call through a register: $(calls vp_turbo); $(grep -m2 'host_gpu' "$work/vp_turbo.err")"; fails=1; }
diff -q "$work/off_vprintf.out" "$work/vp_turbo.out" > /dev/null || { echo "FAIL: turbo mode changes the output of host_gpu_vars printf"; fails=1; }
PROG=vars run vp_exact printf VGPU_HOST_GPU=exact "${reach[@]}" || { echo "FAIL: the exact printf run failed"; fails=1; }
[[ ! -s "$work/vp_exact.log" ]] || { echo "FAIL: exact mode sent a kernel that prints to the card: $(calls vp_exact)"; fails=1; }
PROG=vars run vp_off printf VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TURBO_OFF=printf "${reach[@]}" || { echo "FAIL: the turbo printf run with printf off failed"; fails=1; }
[[ ! -s "$work/vp_off.log" ]] || { echo "FAIL: VGPU_HOST_GPU_TURBO_OFF=printf did not keep the kernel that prints in the simulator: $(calls vp_off)"; fails=1; }
diff -q "$work/off_vprintf.out" "$work/vp_off.out" > /dev/null || { echo "FAIL: VGPU_HOST_GPU_TURBO_OFF=printf changes the output"; fails=1; }
for m in exact turbo; do
  PROG=vars run "vt_$m" tex VGPU_HOST_GPU=$m "${reach[@]}" || { echo "FAIL: the $m tex run failed"; tail -5 "$work/vt_$m.err"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/vt_$m.log" || { echo "FAIL: $m: a live texture object kept a plain kernel off the card: $(calls "vt_$m")"; fails=1; }
  diff -q "$work/off_vtex.out" "$work/vt_$m.out" > /dev/null || { echo "FAIL: $m changes the output of host_gpu_vars tex"; fails=1; }
  PROG=vars run "vh_$m" heap VGPU_HOST_GPU=$m "${reach[@]}" || { echo "FAIL: the $m heap run failed"; tail -5 "$work/vh_$m.err"; fails=1; }
  [[ ! -s "$work/vh_$m.log" ]] || { echo "FAIL: $m: a kernel that calls malloc reached the driver: $(calls "vh_$m")"; fails=1; }
  diff -q "$work/off_vheap.out" "$work/vh_$m.out" > /dev/null || { echo "FAIL: $m changes the output of host_gpu_vars heap"; fails=1; }
done
grep -q '^host_gpu_refused: kernel=[^ ]*fetch[^ ]* class=never insn=T\(EX\|LD\)' "$work/vt_turbo.err" || { echo "FAIL: the texture fetch was not refused as class never: $(grep -m2 '^host_gpu_refused' "$work/vt_turbo.err")"; fails=1; }
# 36. a graph's launches.
for m in exact turbo; do
  PROG=vars run "vg_$m" graph VGPU_HOST_GPU=$m "${reach[@]}" || { echo "FAIL: the $m graph run failed"; tail -5 "$work/vg_$m.err"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/vg_$m.log" || { echo "FAIL: $m: a kernel launched from a graph did not reach the card path: $(calls "vg_$m")"; fails=1; }
  diff -q "$work/off_vgraph.out" "$work/vg_$m.out" > /dev/null || { echo "FAIL: $m changes the output of host_gpu_vars graph"; fails=1; }
done
# 35. a call into another section.
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/rdc" > "$work/off_rdc.out" 2> "$work/off_rdc.err" || { echo "FAIL: host_gpu_rdc with the mode off failed"; tail -5 "$work/off_rdc.err"; exit 1; }
PROG=rdc run rdc_turbo x VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 "${reach[@]}" || { echo "FAIL: the turbo rdc run failed"; tail -5 "$work/rdc_turbo.err"; fails=1; }
grep -q '^cuModuleLoadData$' "$work/rdc_turbo.log" || { echo "FAIL: a kernel that calls a function of another section did not reach the card path in turbo mode: $(calls rdc_turbo); $(grep -m2 'calls_out' "$work/rdc_turbo.err")"; fails=1; }
grep -q '^host_gpu_refused:' "$work/rdc_turbo.err" && { echo "FAIL: turbo refused a kernel of host_gpu_rdc: $(grep -m1 '^host_gpu_refused:' "$work/rdc_turbo.err")"; fails=1; }
diff -q "$work/off_rdc.out" "$work/rdc_turbo.out" > /dev/null || { echo "FAIL: turbo mode changes the output of host_gpu_rdc"; fails=1; }
PROG=rdc run rdc_exact x VGPU_HOST_GPU=exact "${reach[@]}" || { echo "FAIL: the exact rdc run failed"; fails=1; }
[[ ! -s "$work/rdc_exact.log" ]] || { echo "FAIL: exact mode sent a kernel with a call to the card: $(calls rdc_exact)"; fails=1; }
(( fails == 0 )) && { echo "PASS: first launch $first ms against a driver start of $init_ms ms; driver calls in the big run: $(calls on_big)"; exit 0; }
exit 1
