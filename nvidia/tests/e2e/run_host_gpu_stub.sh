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
# What the mode costs a program whose launches stay in the simulator (pantheonsim #662, #660; host_gpu_overhead.cu):
#  45. 50 large kernels launched once each, and one kernel launched 5000 times, in both modes: nothing reaches the driver, the program prints what it prints
#      with the mode off, and the scan says the same of the code as before it was made cheaper (no loop in any of the 50 kernels; a loop in `acct`).
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
# The driver start of a program of many short launches (Round sp69; include/vgpu/host_gpu.hpp start_share_launches):
#  37. `many` launches one kernel 400 times, each launch about a hundredth of the start cost it is given (read from a run whose cost is never repaid):
#      no launch saves the share of the start a launch must save by itself (a 50th), but once the simulator has run the kernel 200 times the start
#      must only be repaid within as many launches again, and the driver starts; with VGPU_HOST_GPU_START_REPEATS=0 (the rule before) it is never
#      touched, nor is it by `manyfew`, the same program with 150 launches (which would repay the start as well, but are too few to say that as many
#      follow); the output is the mode-off output in each.
# Bounded blocks (Round sp75; include/vgpu/host_gpu.hpp kSampleBlockSeconds, vgpu/exec/launch.hpp part_block_limit):
#  38. `slow` launches one kernel once on 256 blocks, each of which takes longer than the whole wait for a sample's copy (the rounds are set from a
#      timed run so that a block takes about 80 ms): the copy stops its first block at the limit and answers (the sample's line says unfinished=1),
#      the launch is sent, and against a stand-in that keeps memory and computes nothing the program prints what `slownone`, which launches nothing,
#      prints;
#  39. ... and with VGPU_HOST_GPU_SAMPLE_BLOCK=0 (the rule before) the copy does not answer, the driver is never touched and the output is the
#      mode-off output;
#  40. `slowfew`, the same work on 96 blocks in a kernel whose loop counts by a parameter (a grid the sample of whole blocks does not take, of
#      a kernel with no loop the sample inside the blocks follows), is sampled by bounded blocks (small=1) and sent, the memory left as it was; with VGPU_HOST_GPU_SAMPLE_BLOCK=0 it is not
#      sampled at all and the driver is never touched.
# A sample once the driver is up (Round sp76; include/vgpu/host_gpu.hpp sample_allowed, src/runtime/host_gpu.cpp sample_plan and sample_copy_begin):
#  41. `two` launches two kernels once each, both on blocks that outlast the wait, beside 64 MiB no kernel touches. The first is sampled and sent
#      and starts the driver; the second is sampled with the driver up (after_driver=1): what only the card held is pulled before the fork, the
#      copy answers, and the launch is sent. Against a stand-in that keeps memory and computes nothing the program prints what `twonone`, which
#      launches nothing, prints;
#  42. ... and with VGPU_HOST_GPU_SAMPLE_STARTED=0 (the rule before) the second kernel is not sampled ("the driver has been started"), the
#      simulator runs it on the memory the card left, and the program prints what `twob`, which launches the second kernel alone, prints;
#  43. `twoshort`, whose second kernel runs two rounds, is sampled with the driver up and kept (its estimate is far under the card path of 64 MiB):
#      one launch reaches the card, and the program prints what `twoshortb` prints;
#  44. against the stand-in that refuses every module, `two` is sampled twice, the second time with the driver up, and prints the mode-off output.
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
# (turbo fit, steps 45 to 52: the same program built for the architectures the stand-in card is told to be, and as PTX only)
if [[ "$cc" == 86 ]]; then
  fitsrc="$root/nvidia/tests/e2e/host_gpu_fit.cu"
  nvf=(-std=c++17 -cudart shared -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim"))
  "$nvcc_bin" "${nvf[@]}" -arch=sm_86 "$fitsrc" -o "$work/fit" || { echo "FAIL: host_gpu_fit.cu does not compile for sm_86"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -arch=sm_90 "$fitsrc" -o "$work/fit90" || { echo "FAIL: host_gpu_fit.cu does not compile for sm_90"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 "$root/nvidia/tests/e2e/host_gpu_vars.cu" -o "$work/vars80" || { echo "FAIL: host_gpu_vars.cu does not compile for sm_80 and sm_86"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -arch=sm_86 "$root/nvidia/tests/e2e/host_gpu_coop.cu" -o "$work/coop" || { echo "FAIL: host_gpu_coop.cu does not compile for sm_86"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -arch=sm_89 "$fitsrc" -o "$work/fit89" || { echo "FAIL: host_gpu_fit.cu does not compile for sm_89"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -DFIT_NO_ELECT -gencode arch=compute_90,code=compute_90 "$fitsrc" -o "$work/fit90p" || { echo "FAIL: host_gpu_fit.cu does not compile as PTX for compute_90"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -arch=sm_80 "$fitsrc" -o "$work/fit80" || { echo "FAIL: host_gpu_fit.cu does not compile for sm_80"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -gencode arch=compute_75,code=compute_75 "$fitsrc" -o "$work/fit75p" || { echo "FAIL: host_gpu_fit.cu does not compile as PTX for compute_75"; exit 1; }
  "$nvcc_bin" "${nvf[@]}" -gencode arch=compute_86,code=compute_86 "$root/nvidia/tests/e2e/host_gpu_vars.cu" -o "$work/varsp" || { echo "FAIL: host_gpu_vars.cu does not compile as PTX"; exit 1; }
fi
"$nvcc_bin" -std=c++17 -arch="sm_$cc" -cudart shared -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$root/nvidia/tests/e2e/host_gpu_overhead.cu" -o "$work/ovh" || { echo "FAIL: host_gpu_overhead.cu does not compile"; exit 1; }
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
# (VGPU_HOST_GPU_FIT=0: steps 10 to 44 pin what turbo mode does with its performance rules on, which is what VGPU_HOST_GPU_FIT=0 keeps; exact mode does not read
# the variable. The default, FIT=1, sends every kernel that fits the card, and is pinned by steps 45 to 52.)
reach=(VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 VGPU_HOST_GPU_FIT=0)
reachx=(VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0)
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
rl='^host_gpu_turbo: mode=turbo launches_card=[0-9]+ launches_sim=[0-9]+ card_s=[0-9.]+ sim_s=[0-9.]+ words_compared=[0-9]+ words_different=[0-9]+ words_outside=[0-9]+ words_subnormal=[0-9]+ kernels_outside=[0-9]+ max_abs=[^ ]+ max_rel=[^ ]+ bits_different=[0-9]+$'
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
str=(VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=0 VGPU_HOST_GPU_FIT=0 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 STUB_INIT_MS=50)   # (FIT=0: the streak rule is one of the performance rules)
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
# (VGPU_HOST_GPU_SHORT=0, the size rule this step was written for: under #660 a launch whose simulator time reaches 40 us on a loaded machine is sent, and 4 to 14 us is the usual figure here)
run str_read streakread VGPU_HOST_GPU=exact VGPU_HOST_GPU_SHORT=0 "${str[@]}" || { echo "FAIL: the run of streakread failed"; fails=1; }
grep -q '^bumps 8 to 8, ' "$work/str_read.out" || { echo "FAIL: a bump went to the card although the program had read the memory back: $(cat "$work/str_read.out")"; fails=1; }
[[ "$(streak_of str_read)" == "0 1" ]] || { echo "FAIL: streakread: $(grep '^host_gpu_streak' "$work/str_read.err")"; fails=1; }
# 27. a zero grid while the card holds the memory (Round sp65: PolyBench LU's last lu_kernel1).
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" zerogrid > "$work/off_zerogrid.out" 2> "$work/off_zerogrid.err" || { echo "FAIL: the zerogrid run with the mode off failed"; exit 1; }
# (cudaErrorInvalidValue: what the A10G answered with the CUDA 13.0 runtime, driver 595.112.07, at the integration of sp64-sp68; sp65 wrote
# cudaErrorInvalidConfiguration here, a CUDA 12 runtime's answer, and the step had not run)
grep -q '^zero grid: cudaErrorInvalidValue$' "$work/off_zerogrid.out" || { echo "FAIL: zerogrid with the mode off: $(cat "$work/off_zerogrid.out")"; fails=1; }
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
# 37. (Round sp69) many short launches of one kernel. The cost: 100 times the shortest launch, so a launch saves at most a 100th of it and the
# simulator has spent it after about 100 launches; from the 200th launch on, the launches so far would repay it if the card path took up to half
# the launch (it takes 0.1 ms by the estimate, so the launch must take 0.3 ms or more: see the check).
for arg in many manyfew; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; tail -5 "$work/off_$arg.err"; exit 1; }
done
run many_never many VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=1000 || { echo "FAIL: the many run with a start cost of 1000 s failed"; fails=1; }
[[ ! -s "$work/many_never.log" ]] || { echo "FAIL: many reached the driver although the start could never be repaid: $(calls many_never)"; fails=1; }
many_us="$(sed -n 's/^min_launch_us=\([0-9]*\).*/\1/p' "$work/many_never.err" | head -1)"
if [[ -n "$many_us" ]] && (( many_us >= 300 )); then
  many_cost="$(awk -v us="$many_us" 'BEGIN { printf "%.6f", us / 10000 }')"
  many=(VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST="$many_cost" STUB_INIT_MS=50)
  run many_run many "${many[@]}" || { echo "FAIL: the many run failed"; tail -5 "$work/many_run.err"; fails=1; }
  grep -q '^cuInit$' "$work/many_run.log" || { echo "FAIL: 400 launches of one kernel, of $many_us us each, did not start a driver that costs $many_cost s: $(calls many_run); $(grep -c 'stays in the simulator' "$work/many_run.err") launches stayed"; fails=1; }
  many_at="$(sed -n 's/.*starts the driver: size rule .* \([0-9]*\) of this kernel).*/\1/p' "$work/many_run.err" | head -1)"
  [[ -n "$many_at" ]] && (( many_at >= 200 && many_at < 400 )) || { echo "FAIL: many started the driver after ${many_at:-?} launches of its kernel, not after 200 to 399: $(grep 'starts the driver' "$work/many_run.err" | head -1)"; fails=1; }
  diff -q "$work/off_many.out" "$work/many_run.out" > /dev/null || { echo "FAIL: the start changes the output of many"; fails=1; }
  run many_old many "${many[@]}" VGPU_HOST_GPU_START_REPEATS=0 || { echo "FAIL: the many run with VGPU_HOST_GPU_START_REPEATS=0 failed"; fails=1; }
  [[ ! -s "$work/many_old.log" ]] || { echo "FAIL: with VGPU_HOST_GPU_START_REPEATS=0 many reached the driver: $(calls many_old)"; fails=1; }
  diff -q "$work/off_many.out" "$work/many_old.out" > /dev/null || { echo "FAIL: VGPU_HOST_GPU_START_REPEATS=0 changes the output of many"; fails=1; }
  run many_few manyfew "${many[@]}" || { echo "FAIL: the manyfew run failed"; fails=1; }
  [[ ! -s "$work/many_few.log" ]] || { echo "FAIL: 150 launches of one kernel reached the driver: $(calls many_few)"; fails=1; }
  diff -q "$work/off_manyfew.out" "$work/many_few.out" > /dev/null || { echo "FAIL: the mode changes the output of manyfew"; fails=1; }
else
  echo "FAIL: the shortest launch of many took ${many_us:-?} us: too short for the test (under 300 us)"; fails=1
fi
# 38 to 40. (Round sp75) bounded blocks. The rounds: from a timed run of `slow` with 1,024, so that one block of 256 threads takes about 80 ms on
# one host thread here (the wait for a copy is 50 ms, the limit of a block 15 ms).
env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" HOST_GPU_LAZY_ROUNDS=1024 "$work/prog" slow > /dev/null 2> "$work/slow_cal.err" || { echo "FAIL: the timed run of slow failed"; tail -5 "$work/slow_cal.err"; exit 1; }
slow_cal_ms="$(sed -n 's/^first_launch_ms=\([0-9]*\).*/\1/p' "$work/slow_cal.err" | head -1)"
slow_rounds="$(awk -v ms="${slow_cal_ms:-0}" 'BEGIN { if (ms < 1) ms = 1; r = int(1024 * 80 * 256 / 4 / ms); if (r < 4096) r = 4096; if (r > 4000000) r = 4000000; print r }')"
export HOST_GPU_LAZY_ROUNDS="$slow_rounds"
for arg in slow slownone slowfew slowfewnone; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; tail -5 "$work/off_$arg.err"; exit 1; }
done
slow_ms="$(sed -n 's/^first_launch_ms=\([0-9]*\).*/\1/p' "$work/off_slow.err" | head -1)"
unfinished_of() { grep -E '^host_gpu_sample: ' "$work/$1.err" | grep -c ' unfinished=1 '; }
small_of() { grep -E '^host_gpu_sample: ' "$work/$1.err" | grep -c ' small=1 '; }
timeouts_of() { sed -n 's/^host_gpu_sampled: .* timeouts=\([0-9]*\) .*/\1/p' "$work/$1.err" | head -1; }
if (( ${slow_ms:-0} * 4 / 256 >= 40 )); then   # (a block takes 40 ms or more: its two runs outlast the wait, and it is stopped at the limit)
  # 38. sent on a block stopped at the limit; the memory the card gets is the memory before the launch.
  # (the copy must be given 30 ms of processor within the 50 ms the process waits: on a busy machine it may not be, and the launch then stays, as
  # before the round. Tried up to three times.)
  for try in 1 2 3; do
    run blk_nop slow "${smp[@]}" STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the run of slow on a card that computes nothing failed"; tail -5 "$work/blk_nop.err"; fails=1; }
    [[ "$(timeouts_of blk_nop)" == 0 ]] && break
  done
  read -r smp_n smp_sent <<< "$(sampled_of blk_nop)"
  (( ${smp_n:-0} == 1 && ${smp_sent:-0} == 1 && $(unfinished_of blk_nop) == 1 )) || { echo "FAIL: slow: expected one sample, of a block stopped at the limit, and the launch sent: $(grep -E 'host_gpu_sampled|host_gpu_sample:|sampled' "$work/blk_nop.err" | head -3)"; fails=1; }
  [[ "$(timeouts_of blk_nop)" == 0 ]] || { echo "FAIL: slow: the copy did not answer although its blocks are bounded: $(grep host_gpu_sampled "$work/blk_nop.err")"; fails=1; }
  diff -q "$work/off_slownone.out" "$work/blk_nop.out" > /dev/null || { echo "FAIL: a launch sent on a block stopped at the limit does not leave the memory as it was before the launch"; fails=1; }
  diff -q "$work/off_slow.out" "$work/blk_nop.out" > /dev/null && { echo "FAIL: the run of slow on a card that computes nothing printed the simulator's result: nothing was sent"; fails=1; }
  # 39. the rule before: the copy runs its first block whole, does not answer in time, and the launch is the simulator's.
  run blk_old slow "${smp[@]}" VGPU_HOST_GPU_SAMPLE_BLOCK=0 || { echo "FAIL: the run of slow with VGPU_HOST_GPU_SAMPLE_BLOCK=0 failed"; tail -5 "$work/blk_old.err"; fails=1; }
  read -r smp_n smp_sent <<< "$(sampled_of blk_old)"
  (( ${smp_n:-0} == 1 && ${smp_sent:-1} == 0 )) && [[ "$(timeouts_of blk_old)" == 1 ]] || { echo "FAIL: slow with VGPU_HOST_GPU_SAMPLE_BLOCK=0: expected one sample whose copy did not answer: $(grep host_gpu_sampled "$work/blk_old.err")"; fails=1; }
  [[ ! -s "$work/blk_old.log" ]] || { echo "FAIL: slow with VGPU_HOST_GPU_SAMPLE_BLOCK=0 reached the driver: $(calls blk_old)"; fails=1; }
  diff -q "$work/off_slow.out" "$work/blk_old.out" > /dev/null || { echo "FAIL: VGPU_HOST_GPU_SAMPLE_BLOCK=0 changes the output of slow"; fails=1; }
  # 40. a grid of 96 blocks with no followed loop: sampled by bounded blocks and sent; not sampled under the rule before.
  run few_nop slowfew "${smp[@]}" VGPU_HOST_GPU_START_COST=0 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the run of slowfew on a card that computes nothing failed"; tail -5 "$work/few_nop.err"; fails=1; }
  read -r smp_n smp_sent <<< "$(sampled_of few_nop)"
  (( ${smp_n:-0} == 1 && ${smp_sent:-0} == 1 && $(small_of few_nop) == 1 && $(unfinished_of few_nop) == 1 )) || { echo "FAIL: slowfew: expected one sample by bounded blocks of a small grid, and the launch sent: $(grep -E 'host_gpu_sampled|host_gpu_sample:|sampled' "$work/few_nop.err" | head -3)"; fails=1; }
  diff -q "$work/off_slowfewnone.out" "$work/few_nop.out" > /dev/null || { echo "FAIL: a launch of a small grid sent on a stopped block does not leave the memory as it was before the launch"; fails=1; }
  run few_old slowfew "${smp[@]}" VGPU_HOST_GPU_START_COST=0 VGPU_HOST_GPU_SAMPLE_BLOCK=0 || { echo "FAIL: the run of slowfew with VGPU_HOST_GPU_SAMPLE_BLOCK=0 failed"; tail -5 "$work/few_old.err"; fails=1; }
  read -r smp_n smp_sent <<< "$(sampled_of few_old)"
  [[ "${smp_n:-x}" == 0 ]] || { echo "FAIL: slowfew with VGPU_HOST_GPU_SAMPLE_BLOCK=0 was sampled: $(grep host_gpu_sampled "$work/few_old.err")"; fails=1; }
  [[ ! -s "$work/few_old.log" ]] || { echo "FAIL: slowfew with VGPU_HOST_GPU_SAMPLE_BLOCK=0 reached the driver: $(calls few_old)"; fails=1; }
  diff -q "$work/off_slowfew.out" "$work/few_old.out" > /dev/null || { echo "FAIL: VGPU_HOST_GPU_SAMPLE_BLOCK=0 changes the output of slowfew"; fails=1; }
  # 41 to 44. (Round sp76) a kernel's first launch once the driver is up. (The start costs nothing here, so that the first launch is sent whatever
  # the 64 MiB add to its card path; the second launch has no start to repay either way.)
  for arg in two twonone twob twoshort twoshortb; do
    env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/prog" "$arg" > "$work/off_$arg.out" 2> "$work/off_$arg.err" || { echo "FAIL: the $arg run with the mode off failed"; tail -5 "$work/off_$arg.err"; exit 1; }
  done
  after_of() { sed -n 's/^host_gpu_sampled: .* after_driver=\([0-9]*\) .*/\1/p' "$work/$1.err" | head -1; }
  launches_of() { grep -c '^cuLaunchKernel$' "$work/$1.log"; }
  up=("${smp[@]}" VGPU_HOST_GPU_START_COST=0)
  # 41. both launches sent, the second on a sample taken with the driver up.
  for try in 1 2 3; do
    run two_nop two "${up[@]}" STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the run of two on a card that computes nothing failed"; tail -5 "$work/two_nop.err"; fails=1; }
    [[ "$(timeouts_of two_nop)" == 0 ]] && break
  done
  read -r smp_n smp_sent <<< "$(sampled_of two_nop)"
  (( ${smp_n:-0} == 2 && ${smp_sent:-0} == 2 )) && [[ "$(after_of two_nop)" == 1 && "$(launches_of two_nop)" == 2 ]] || { echo "FAIL: two: expected two samples, the second with the driver up, and both launches sent: $(grep -E 'host_gpu_sampled|host_gpu_nosample|sampled' "$work/two_nop.err" | head -4); $(calls two_nop)"; fails=1; }
  grep -q 'MiB only the card held pulled before a sample' "$work/two_nop.err" || { echo "FAIL: two: nothing was pulled before the sample taken with the driver up, although the card held the memory"; fails=1; }
  diff -q "$work/off_twonone.out" "$work/two_nop.out" > /dev/null || { echo "FAIL: two launches sent to a card that computes nothing do not leave the memory as it was"; fails=1; }
  diff -q "$work/off_two.out" "$work/two_nop.out" > /dev/null && { echo "FAIL: the run of two on a card that computes nothing printed the simulator's result"; fails=1; }
  # 42. the rule before: no sample once the driver has been started; the simulator runs the second launch on what the card left.
  for try in 1 2 3; do
    run two_old two "${up[@]}" VGPU_HOST_GPU_SAMPLE_STARTED=0 STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the run of two with VGPU_HOST_GPU_SAMPLE_STARTED=0 failed"; tail -5 "$work/two_old.err"; fails=1; }
    [[ "$(timeouts_of two_old)" == 0 ]] && break
  done
  read -r smp_n smp_sent <<< "$(sampled_of two_old)"
  (( ${smp_n:-0} == 1 && ${smp_sent:-0} == 1 )) && [[ "$(after_of two_old)" == 0 && "$(launches_of two_old)" == 1 ]] || { echo "FAIL: two with VGPU_HOST_GPU_SAMPLE_STARTED=0: expected one sample and one launch sent: $(grep -E 'host_gpu_sampled|host_gpu_nosample' "$work/two_old.err" | head -3); $(calls two_old)"; fails=1; }
  grep -q '^host_gpu_nosample: .*reason=the driver has been started' "$work/two_old.err" || { echo "FAIL: two with VGPU_HOST_GPU_SAMPLE_STARTED=0: the second kernel was not kept for the driver's start: $(grep '^host_gpu_nosample' "$work/two_old.err" | head -2)"; fails=1; }
  diff -q "$work/off_twob.out" "$work/two_old.out" > /dev/null || { echo "FAIL: with VGPU_HOST_GPU_SAMPLE_STARTED=0 the second launch of two did not run in the simulator on the memory the card left"; fails=1; }
  # 43. a short second launch: sampled with the driver up, kept, run by the simulator on what the card left.
  for try in 1 2 3; do
    run two_short twoshort "${up[@]}" STUB_CARD_OK=1 STUB_KEEP_MEMORY=1 || { echo "FAIL: the run of twoshort failed"; tail -5 "$work/two_short.err"; fails=1; }
    [[ "$(timeouts_of two_short)" == 0 ]] && break
  done
  read -r smp_n smp_sent <<< "$(sampled_of two_short)"
  (( ${smp_n:-0} == 2 && ${smp_sent:-0} == 1 )) && [[ "$(after_of two_short)" == 1 && "$(launches_of two_short)" == 1 ]] || { echo "FAIL: twoshort: expected two samples, the second with the driver up and kept: $(grep -E 'host_gpu_sampled|host_gpu_nosample|sampled' "$work/two_short.err" | head -4); $(calls two_short)"; fails=1; }
  diff -q "$work/off_twoshortb.out" "$work/two_short.out" > /dev/null || { echo "FAIL: a launch sampled with the driver up and kept did not run in the simulator on the memory the card left"; fails=1; }
  # 44. a driver that refuses every module: both launches come back to the simulator, and the output is the mode-off output.
  for try in 1 2 3; do
    run two_ref two "${up[@]}" || { echo "FAIL: the run of two against a driver that refuses modules failed"; tail -5 "$work/two_ref.err"; fails=1; }
    [[ "$(timeouts_of two_ref)" == 0 ]] && break
  done
  read -r smp_n smp_sent <<< "$(sampled_of two_ref)"
  (( ${smp_n:-0} == 2 )) && [[ "$(after_of two_ref)" == 1 ]] || { echo "FAIL: two against a driver that refuses modules: expected two samples, the second with the driver up: $(grep -E 'host_gpu_sampled|host_gpu_nosample' "$work/two_ref.err" | head -3)"; fails=1; }
  diff -q "$work/off_two.out" "$work/two_ref.out" > /dev/null || { echo "FAIL: a sample taken with the driver up changes the output of two"; fails=1; }
else
  echo "FAIL: a block of slow took $(( ${slow_ms:-0} * 4 / 256 )) ms with $slow_rounds rounds (a timed run of 1,024 took ${slow_cal_ms:-?} ms): too short for the test (under 40 ms)"; fails=1
fi
unset HOST_GPU_LAZY_ROUNDS
# ---- turbo fit (docs/perf-design/12-host-gpu-turbo.md "Turbo fit") ----
# Turbo mode sends every kernel the card can run (VGPU_HOST_GPU_FIT=1, the default), from the image the CARD runs, whatever the simulated profile is. The stand-in
# card is told what it is (STUB_SM, STUB_MP) and what its limits are (STUB_MAX_THREADS, STUB_SHARED_OPTIN, STUB_MAX_GRID_X, STUB_FUNC_REGS ...).
#  45. a launch of one warp, far under the size rule, goes to the card in turbo mode, and stays in the simulator with VGPU_HOST_GPU_FIT=0 (and in exact mode);
#      a value of the switch that is neither 0 nor 1 turns the mode off with a message;
#  46. a launch that does not fit the card stays in the simulator and the mode goes on: more threads than the card allows a block, than the kernel can run, more
#      registers for the block than the card has, more shared memory, a longer grid, more simulated memory than the card has; the same launch with the card's
#      limits wide enough is sent;
#  47. a kernel with an instruction no turbo class names (ELECT, sm_90) is sent in turbo mode, not in exact mode, and not with
#      VGPU_HOST_GPU_TURBO_OFF=unlisted (class=off);
#  48. a profile of another architecture than the card's (a simulated H100 on an sm_86 card): turbo mode runs, on the card's own cubin (the stand-in says which
#      image it was given); exact mode is off, with its message;
#  49. the image follows the card: sm_86 and sm_80 cubins for an sm_86 card, PTX when no cubin of the card's major runs (an sm_90 card given an sm_86 build),
#      nothing when no image runs (an sm_80 card given an sm_86 build, an sm_86 card given an sm_90 build);
#  50. PTX is given to the card only when its module holds no variable;
#  51. a launch the card refuses after the fit check says so, the simulator runs it, and the output is the mode-off output;
#  52. turbo mode says once that its results may differ from the simulator's; exact mode says nothing of the kind.
if [[ "$cc" == 86 ]]; then
  fitoff() { env VGPU_GPU="${2:-$profile}" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/${PROG:-fit}" "$1" > "$work/off_f_$1.out" 2> "$work/off_f_$1.err" || { echo "FAIL: the $1 run of ${PROG:-fit} with the mode off failed"; tail -5 "$work/off_f_$1.err"; exit 1; }; }
  fitline() { sed -n "s/^host_gpu_fit:\(.*\) $2=\([0-9a-z]*\).*/\2/p" "$work/$1.err" | tail -1; }
  for arg in tiny wide shared grid mem; do fitoff "$arg"; done
  # 45.
  PROG=fit run fit_tiny tiny VGPU_HOST_GPU=turbo || { echo "FAIL: the turbo tiny run failed"; tail -5 "$work/fit_tiny.err"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/fit_tiny.log" || { echo "FAIL: a launch of one warp did not reach the card in turbo mode: $(calls fit_tiny); $(grep -m2 host_gpu "$work/fit_tiny.err")"; fails=1; }
  diff -q "$work/off_f_tiny.out" "$work/fit_tiny.out" > /dev/null || { echo "FAIL: turbo mode with a card that refuses the module changes the output of fit tiny"; fails=1; }
  [[ "$(fitline fit_tiny policy)" == fit ]] || { echo "FAIL: no 'host_gpu_fit: policy=fit' line: $(grep '^host_gpu_fit' "$work/fit_tiny.err")"; fails=1; }
  PROG=fit run fit_rules tiny VGPU_HOST_GPU=turbo VGPU_HOST_GPU_FIT=0 || { echo "FAIL: the turbo tiny run with VGPU_HOST_GPU_FIT=0 failed"; fails=1; }
  [[ ! -s "$work/fit_rules.log" ]] || { echo "FAIL: VGPU_HOST_GPU_FIT=0 sent a launch of one warp to the card: $(calls fit_rules)"; fails=1; }
  [[ "$(fitline fit_rules policy)" == rules ]] || { echo "FAIL: no 'host_gpu_fit: policy=rules' line with VGPU_HOST_GPU_FIT=0: $(grep '^host_gpu_fit' "$work/fit_rules.err")"; fails=1; }
  PROG=fit run fit_exact tiny VGPU_HOST_GPU=exact || { echo "FAIL: the exact tiny run failed"; fails=1; }
  [[ ! -s "$work/fit_exact.log" ]] || { echo "FAIL: exact mode sent a launch of one warp to the card: $(calls fit_exact)"; fails=1; }
  PROG=fit run fit_badsw tiny VGPU_HOST_GPU=turbo VGPU_HOST_GPU_FIT=2 || { echo "FAIL: the run with VGPU_HOST_GPU_FIT=2 failed"; fails=1; }
  [[ ! -s "$work/fit_badsw.log" ]] || { echo "FAIL: VGPU_HOST_GPU_FIT=2 reached the driver: $(calls fit_badsw)"; fails=1; }
  [[ "$(grep -c 'VGPU_HOST_GPU_FIT=2 is not 0 or 1; mode off' "$work/fit_badsw.err")" == 1 ]] || { echo "FAIL: VGPU_HOST_GPU_FIT=2 was not said once to be bad: $(grep -m2 host_gpu "$work/fit_badsw.err")"; fails=1; }
  run fit_exact_sw big VGPU_HOST_GPU=exact VGPU_HOST_GPU_FIT=2 "${reachx[@]}" || { echo "FAIL: the exact run with VGPU_HOST_GPU_FIT=2 failed"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/fit_exact_sw.log" || { echo "FAIL: exact mode was switched off by VGPU_HOST_GPU_FIT (a turbo-only variable)"; fails=1; }
  # 46.
  lim() {   # <tag> <program argument> <expected: refused|sent> [env...]
    local tag=$1 arg=$2 want=$3
    shift 3
    if [[ $want == refused ]]; then
      PROG=fit run "$tag" "$arg" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 STUB_CARD_OK=1 "$@" || { echo "FAIL: the $tag run failed"; tail -5 "$work/$tag.err"; fails=1; }
      ! grep -q '^cuLaunchKernel$' "$work/$tag.log" || { echo "FAIL: $tag: a launch that does not fit the card was sent to it: $(calls "$tag")"; fails=1; }
      (( $(fitline "$tag" refused_fit) >= 1 )) || { echo "FAIL: $tag: refused_fit is not counted: $(grep '^host_gpu_fit' "$work/$tag.err")"; fails=1; }
      grep -q 'does not fit the card' "$work/$tag.err" || { echo "FAIL: $tag: the log does not say that the launch does not fit: $(grep -m2 host_gpu "$work/$tag.err")"; fails=1; }
      grep -q 'failed on the card (error\|; mode off' "$work/$tag.err" && { echo "FAIL: $tag: a launch that does not fit ended the mode: $(grep -m1 'failed on the card (error\|; mode off' "$work/$tag.err")"; fails=1; }
      diff -q "$work/off_f_$arg.out" "$work/$tag.out" > /dev/null || { echo "FAIL: $tag changes the output of fit $arg"; fails=1; }
    else   # (a verify run: the card computes nothing, the program's output is the simulator's)
      PROG=fit run "$tag" "$arg" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 "$@" || { echo "FAIL: the $tag run failed"; tail -5 "$work/$tag.err"; fails=1; }
      grep -q '^cuLaunchKernel$' "$work/$tag.log" || { echo "FAIL: $tag: a launch that fits the card was not sent: $(calls "$tag"); $(grep -m2 host_gpu "$work/$tag.err")"; fails=1; }
      (( $(fitline "$tag" refused_fit) == 0 )) || { echo "FAIL: $tag: a launch that fits was counted as not fitting"; fails=1; }
      diff -q "$work/off_f_$arg.out" "$work/$tag.out" > /dev/null || { echo "FAIL: $tag changes the output of fit $arg"; fails=1; }
    fi
  }
  lim fl_threads wide refused STUB_MAX_THREADS=512
  lim fl_threads_ok wide sent
  lim fl_blockx wide refused STUB_MAX_BLOCK_X=512
  lim fl_kernel wide refused STUB_FUNC_MAX_THREADS=512
  lim fl_regs wide refused STUB_FUNC_REGS=100
  lim fl_regs_ok wide sent STUB_FUNC_REGS=64
  lim fl_shared shared refused STUB_SHARED_OPTIN=49152
  lim fl_shared_ok shared sent STUB_SHARED_OPTIN=65536
  lim fl_shared_static shared refused STUB_FUNC_SHARED=40000
  lim fl_grid grid refused STUB_MAX_GRID_X=1000
  lim fl_grid_ok grid sent STUB_MAX_GRID_X=4096
  lim fl_mem mem refused STUB_TOTAL_MIB=4
  lim fl_mem_ok mem sent STUB_TOTAL_MIB=4096
  # 46b. a cooperative launch goes to the card as one, when the card can hold the grid at once (#661); VGPU_HOST_GPU_COOP=0 and exact mode keep it
  # in the simulator. A launch with an argument into mapped host memory goes with the memory registered at its own address; VGPU_HOST_GPU_HOSTMAP=0,
  # exact mode, a registration the driver refuses, and a device pointer that is not the host pointer keep it in the simulator.
  PROG=coop fitoff coop
  fitoff hostmap
  turbo_field() { grep -m1 '^host_gpu_turbo:' "$work/$1.err" | grep -o "$2=[0-9]*" | cut -d= -f2; }
  coop_case() {   # <tag> <sent|kept> [env...]
    local tag=$1 want=$2
    shift 2
    PROG=coop run "$tag" coop VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 "$@" || { echo "FAIL: the $tag run failed"; tail -5 "$work/$tag.err"; fails=1; }
    ! grep -q '^cuLaunchKernel$' "$work/$tag.log" || { echo "FAIL: $tag: a cooperative launch was made with cuLaunchKernel: $(calls "$tag")"; fails=1; }
    if [[ $want == sent ]]; then
      grep -q '^cuLaunchCooperativeKernel$' "$work/$tag.log" || { echo "FAIL: $tag: a cooperative launch that fits the card was not sent: $(calls "$tag"); $(grep -m2 host_gpu "$work/$tag.err")"; fails=1; }
    else
      ! grep -q '^cuLaunchCooperativeKernel$' "$work/$tag.log" || { echo "FAIL: $tag: a cooperative launch that must stay was sent: $(calls "$tag")"; fails=1; }
    fi
    grep -q 'launch=0' "$work/$tag.out" || { echo "FAIL: $tag: the simulator did not run the launch: $(cat "$work/$tag.out")"; fails=1; }
    grep -q '; mode off\|failed on the card (error' "$work/$tag.err" && { echo "FAIL: $tag: the mode ended: $(grep -m1 '; mode off\|failed on the card (error' "$work/$tag.err")"; fails=1; }
  }
  coop_case co_sent sent
  coop_case co_off kept VGPU_HOST_GPU_COOP=0
  coop_case co_exact kept VGPU_HOST_GPU=exact
  coop_case co_fits sent FIT_COOP_BLOCKS=40 STUB_OCC=2
  coop_case co_big kept FIT_COOP_BLOCKS=40 STUB_OCC=1
  coop_case co_nocoop kept STUB_COOP=0
  grep -q 'cooperative grid that the card cannot hold' "$work/co_big.err" || { echo "FAIL: co_big: the log does not say the grid is too large for the card: $(grep -m2 host_gpu "$work/co_big.err")"; fails=1; }
  (( $(fitline co_big refused_fit) >= 1 )) || { echo "FAIL: co_big: refused_fit is not counted"; fails=1; }
  hostmap_case() {   # <tag> <sent|kept> [env...]
    local tag=$1 want=$2
    shift 2
    PROG=fit run "$tag" hostmap VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 "$@" || { echo "FAIL: the $tag run failed"; tail -5 "$work/$tag.err"; fails=1; }
    diff -q "$work/off_f_hostmap.out" "$work/$tag.out" > /dev/null || { echo "FAIL: $tag changes the output of fit hostmap: $(cat "$work/$tag.out")"; fails=1; }
    if [[ $want == sent ]]; then
      grep -q '^cuLaunchKernel$' "$work/$tag.log" && grep -q '^cuMemHostRegister$' "$work/$tag.log" || { echo "FAIL: $tag: a launch with a mapped host argument was not sent with its memory registered: $(calls "$tag"); $(grep -m2 host_gpu "$work/$tag.err")"; fails=1; }
      # (the stand-in card computes nothing, so the card's bytes of the host buffer are zeros and the simulator's are not: the compare must have seen the buffer)
      (( $(turbo_field "$tag" words_different) >= 32 )) || { echo "FAIL: $tag: the verify compare did not include the mapped host buffer: $(grep -m1 '^host_gpu_turbo' "$work/$tag.err")"; fails=1; }
    else
      ! grep -q '^cuLaunchKernel$' "$work/$tag.log" || { echo "FAIL: $tag: a launch with a mapped host argument that must stay was sent: $(calls "$tag")"; fails=1; }
    fi
    grep -q '; mode off\|failed on the card (error' "$work/$tag.err" && { echo "FAIL: $tag: the mode ended: $(grep -m1 '; mode off\|failed on the card (error' "$work/$tag.err")"; fails=1; }
  }
  hostmap_case hm_sent sent
  hostmap_case hm_off kept VGPU_HOST_GPU_HOSTMAP=0
  ! grep -q '^cuMemHostRegister$' "$work/hm_off.log" || { echo "FAIL: hm_off: memory was registered with the switch off"; fails=1; }
  hostmap_case hm_exact kept VGPU_HOST_GPU=exact
  hostmap_case hm_regfail kept STUB_HOSTREG_FAIL=1
  (( $(fitline hm_regfail refused_fit) >= 1 )) && grep -q 'mapped host memory at its own address' "$work/hm_regfail.err" || { echo "FAIL: hm_regfail: the refusal is not counted and said: $(grep -m2 host_gpu "$work/hm_regfail.err")"; fails=1; }
  hostmap_case hm_shift kept STUB_HOSTPTR_SHIFT=4096
  grep -q '^cuMemHostUnregister$' "$work/hm_shift.log" || { echo "FAIL: hm_shift: a registration whose device pointer is not the host pointer was kept: $(calls hm_shift)"; fails=1; }
  # 47.
  # (QMMA, FP8 tensor cores, sm_89: no class names it, and a card that has it runs it)
  PROG=fit89 fitoff qmma nvidia/rtx4090
  for m in turbo exact; do
    PROG=fit89 run "fe_$m" qmma VGPU_GPU=nvidia/rtx4090 STUB_SM=89 STUB_MP=128 VGPU_HOST_GPU=$m VGPU_HOST_GPU_LOG=1 "${reachx[@]}" || { echo "FAIL: the $m qmma run failed"; tail -5 "$work/fe_$m.err"; fails=1; }
    diff -q "$work/off_f_qmma.out" "$work/fe_$m.out" > /dev/null || { echo "FAIL: $m changes the output of fit89 qmma"; fails=1; }
  done
  grep -q '^cuModuleLoadData$' "$work/fe_turbo.log" && grep -q '^image elf sm_89$' "$work/fe_turbo.log" || { echo "FAIL: a kernel with an instruction no class names did not reach the card in turbo mode: $(calls fe_turbo); $(grep -m3 host_gpu "$work/fe_turbo.err")"; fails=1; }
  grep -q '^host_gpu_refused:' "$work/fe_turbo.err" && { echo "FAIL: turbo refused a kernel with an unlisted instruction: $(grep -m1 '^host_gpu_refused' "$work/fe_turbo.err")"; fails=1; }
  grep -q 'classes: .*unlisted' "$work/fe_turbo.err" || { echo "FAIL: the kernel's classes do not include unlisted: $(grep -m2 'classes:' "$work/fe_turbo.err")"; fails=1; }
  ! grep -q '^cuModuleLoadData$' "$work/fe_exact.log" || { echo "FAIL: exact mode sent a kernel with an unlisted instruction to the card: $(calls fe_exact)"; fails=1; }
  PROG=fit89 run fe_off qmma VGPU_GPU=nvidia/rtx4090 STUB_SM=89 STUB_MP=128 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_TURBO_OFF=unlisted "${reachx[@]}" || { echo "FAIL: the qmma run with unlisted off failed"; fails=1; }
  ! grep -q '^cuModuleLoadData$' "$work/fe_off.log" || { echo "FAIL: VGPU_HOST_GPU_TURBO_OFF=unlisted did not keep the kernel in the simulator: $(calls fe_off)"; fails=1; }
  grep -q '^host_gpu_refused: kernel=[^ ]* class=off insn=QMMA' "$work/fe_off.err" || { echo "FAIL: no class=off line for QMMA: $(grep -m2 '^host_gpu_refused' "$work/fe_off.err")"; fails=1; }
  # (ELECT, sm_90: a form no card run has shown safe: class never in turbo, exact refuses it too)
  PROG=fit90 fitoff elect nvidia/h100
  grep -q '^mode=elect sum=1$' "$work/off_f_elect.out" || { echo "FAIL: the sm_90 build of fit elect printed: $(cat "$work/off_f_elect.out")"; fails=1; }
  PROG=fit90 run fe_elect elect VGPU_GPU=nvidia/h100 STUB_SM=90 STUB_MP=132 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 "${reachx[@]}" || { echo "FAIL: the turbo elect run failed"; fails=1; }
  ! grep -q '^cuModuleLoadData$' "$work/fe_elect.log" && grep -q '^host_gpu_refused: kernel=[^ ]* class=never insn=ELECT' "$work/fe_elect.err" || { echo "FAIL: ELECT (an sm_90 form no card run has shown safe) was not refused as class never: $(calls fe_elect); $(grep -m2 '^host_gpu_refused' "$work/fe_elect.err")"; fails=1; }
  diff -q "$work/off_f_elect.out" "$work/fe_elect.out" > /dev/null || { echo "FAIL: the elect refusal changes the output"; fails=1; }
  # 48. a simulated H100 (sm_90) on an sm_86 card: turbo mode runs the card's own sm_86 cubin; exact mode is off.
  fitoff tiny nvidia/h100
  PROG=fit run h_turbo tiny VGPU_GPU=nvidia/h100 STUB_SM=86 STUB_MP=28 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the turbo run of an H100 profile on an sm_86 card failed"; tail -5 "$work/h_turbo.err"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/h_turbo.log" && grep -q '^image elf sm_86$' "$work/h_turbo.log" || { echo "FAIL: an H100 profile on an sm_86 card did not give the card its sm_86 cubin: $(calls h_turbo); $(grep -m3 host_gpu "$work/h_turbo.err")"; fails=1; }
  grep -q 'is not the simulated device\|mode off' "$work/h_turbo.err" && { echo "FAIL: turbo mode went off for a profile of another architecture: $(grep -m1 'mode off' "$work/h_turbo.err")"; fails=1; }
  [[ "$(fitline h_turbo card_sm)" == 86 && "$(fitline h_turbo card_sms)" == 28 ]] || { echo "FAIL: the card's architecture is not in the host_gpu_fit line: $(grep '^host_gpu_fit' "$work/h_turbo.err")"; fails=1; }
  diff -q "$work/off_f_tiny.out" "$work/h_turbo.out" > /dev/null || { echo "FAIL: turbo mode changes the output of fit tiny under an H100 profile"; fails=1; }
  # ... the same two gates in exact mode, which keeps them: a card of another architecture, and a card with another SM count, than the profile's
  run gate_sm big VGPU_HOST_GPU=exact STUB_SM=90 STUB_MP=132 "${reachx[@]}" || { echo "FAIL: the exact run on an sm_90 card failed"; fails=1; }
  grep -q 'the card (sm_90, 132 SMs) is not the simulated device (sm_86, 28 SMs); mode off' "$work/gate_sm.err" && ! grep -q '^cuModuleLoadData$' "$work/gate_sm.log" || { echo "FAIL: exact mode ran on a card of another architecture than the profile's: $(calls gate_sm); $(grep -m2 host_gpu "$work/gate_sm.err")"; fails=1; }
  run gate_mp big VGPU_HOST_GPU=exact STUB_MP=80 "${reachx[@]}" || { echo "FAIL: the exact run on a card of 80 SMs failed"; fails=1; }
  grep -q 'the card (sm_86, 80 SMs) is not the simulated device (sm_86, 28 SMs); mode off' "$work/gate_mp.err" && ! grep -q '^cuModuleLoadData$' "$work/gate_mp.log" || { echo "FAIL: exact mode ran on a card with another SM count than the profile's: $(calls gate_mp); $(grep -m2 host_gpu "$work/gate_mp.err")"; fails=1; }
  # ... and turbo mode runs on both (the program is `big`: launches that the performance rules send)
  run gate_t_sm big VGPU_HOST_GPU=turbo STUB_SM=90 STUB_MP=132 "${reachx[@]}" || { echo "FAIL: the turbo run on an sm_90 card failed"; fails=1; }
  grep -q 'is not the simulated device\|; mode off' "$work/gate_t_sm.err" && { echo "FAIL: turbo mode went off for a card of another architecture: $(grep -m1 'mode off' "$work/gate_t_sm.err")"; fails=1; }
  grep -q '^image ptx$' "$work/gate_t_sm.log" || { echo "FAIL: an sm_90 card was not given the PTX of an sm_86 build: $(calls gate_t_sm); $(grep -m2 host_gpu "$work/gate_t_sm.err")"; fails=1; }
  run gate_t_mp big VGPU_HOST_GPU=turbo STUB_MP=80 "${reachx[@]}" || { echo "FAIL: the turbo run on a card of 80 SMs failed"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/gate_t_mp.log" && ! grep -q '; mode off' "$work/gate_t_mp.err" || { echo "FAIL: turbo mode did not run on a card with another SM count than the profile's: $(calls gate_t_mp); $(grep -m2 host_gpu "$work/gate_t_mp.err")"; fails=1; }
  for t in gate_sm gate_mp gate_t_sm gate_t_mp; do diff -q "$work/off_big.out" "$work/$t.out" > /dev/null || { echo "FAIL: $t changes the output of big"; fails=1; }; done
  # 49. the image follows the card.
  img() {   # <tag> <program> <card sm> <expected image line, or none> [env...]
    local tag=$1 prog=$2 sm=$3 want=$4
    shift 4
    PROG=$prog run "$tag" tiny STUB_SM="$sm" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 "$@" || { echo "FAIL: the $tag run failed"; tail -5 "$work/$tag.err"; fails=1; }
    diff -q "$work/off_f_tiny.out" "$work/$tag.out" > /dev/null || { echo "FAIL: $tag changes the output of tiny"; fails=1; }
    if [[ $want == none ]]; then
      ! grep -q '^cuModuleLoadData$' "$work/$tag.log" || { echo "FAIL: $tag: an image that does not run on sm_$sm was given to the card: $(calls "$tag")"; fails=1; }
      (( $(fitline "$tag" refused_image) >= 1 )) || { echo "FAIL: $tag: no image refusal counted: $(grep '^host_gpu_fit' "$work/$tag.err")"; fails=1; }
      grep -q 'no image of the module runs on the card' "$work/$tag.err" || { echo "FAIL: $tag: the reason is not said: $(grep -m2 host_gpu "$work/$tag.err")"; fails=1; }
    else
      grep -q "^$want\$" "$work/$tag.log" || { echo "FAIL: $tag: the card was not given '$want': $(grep '^image' "$work/$tag.log" | head -2)"; fails=1; }
    fi
  }
  img im_86 fit 86 "image elf sm_86"
  img im_80 fit80 86 "image elf sm_80"                      # the same major, a lower minor
  img im_ptx fit 90 "image ptx" STUB_MP=132                   # an sm_90 card: the sm_86 cubin does not run, its PTX does
  img im_75p fit75p 86 "image ptx"                            # a build that holds PTX for compute_75 only: the driver compiles it
  img im_none fit 80 none STUB_MP=108                         # an sm_80 card, an sm_86 build: no cubin and no PTX that runs
  fitoff tiny nvidia/h100
  PROG=fit90 fitoff tiny nvidia/h100
  img_h() { PROG=fit90 run "$1" tiny VGPU_GPU=nvidia/h100 STUB_SM="$2" STUB_MP=28 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the $1 run failed"; fails=1; }; }
  # (the sm_90 build holds an ELECT kernel, so its PTX is refused for the card; the reason is the form, not a missing image)
  img_h im_90_on_86 86
  ! grep -q '^cuModuleLoadData$' "$work/im_90_on_86.log" && grep -q 'compute_90 PTX could be retargeted to sm_86 but the PTX uses an sm_90 or later form (elect' "$work/im_90_on_86.err" || { echo "FAIL: an sm_90 build with an ELECT kernel reached an sm_86 card: $(calls im_90_on_86); $(grep -m2 host_gpu "$work/im_90_on_86.err")"; fails=1; }
  diff -q "$work/off_f_tiny.out" "$work/im_90_on_86.out" > /dev/null || { echo "FAIL: the sm_90 build under an H100 profile changes its output on an sm_86 card"; fails=1; }
  # ... and the one without it (PTX for compute_90 only) is retargeted: the driver is given `.target sm_86`, and the card path is reached; not on an sm_90 card (the PTX is its own),
  # not on sm_75 or sm_89 (the rule is for sm_80, 86 and 87), and VGPU_HOST_GPU_FIT=0 does the same (the rule is the image's, not the size rule's)
  PROG=fit90p fitoff tiny nvidia/h100
  rt90() { PROG=fit90p run "$1" tiny VGPU_GPU=nvidia/h100 STUB_SM="$2" STUB_MP=28 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 STUB_CARD_OK=1 VGPU_HOST_GPU_VERIFY=1 "${@:3}" || { echo "FAIL: the $1 run failed"; fails=1; }; }
  rt90 rt_86 86
  grep -q '^image ptx$' "$work/rt_86.log" && grep -q '^ptx_target sm_86$' "$work/rt_86.log" || { echo "FAIL: plain sm_90 PTX was not given to an sm_86 card with its target rewritten: $(grep -E '^image|^ptx_target' "$work/rt_86.log" | tr '\n' ' ')"; fails=1; }
  [[ "$(fitline rt_86 retargeted)" == 1 ]] || { echo "FAIL: no retargeted module counted: $(grep '^host_gpu_fit' "$work/rt_86.err")"; fails=1; }
  diff -q "$work/off_f_tiny.out" "$work/rt_86.out" > /dev/null || { echo "FAIL: a retargeted module changes the output"; fails=1; }
  rt90 rt_80 80
  grep -q '^ptx_target sm_80$' "$work/rt_80.log" || { echo "FAIL: not retargeted to an sm_80 card: $(grep -E '^image|^ptx_target' "$work/rt_80.log" | tr '\n' ' ')"; fails=1; }
  rt90 rt_90 90 STUB_MP=132
  grep -q '^ptx_target sm_90$' "$work/rt_90.log" && [[ "$(fitline rt_90 retargeted)" == 0 ]] || { echo "FAIL: an sm_90 card was given a retargeted module: $(grep -E '^image|^ptx_target' "$work/rt_90.log" | tr '\n' ' ')"; fails=1; }
  for sm in 75 89; do
    rt90 rt_$sm $sm
    ! grep -q '^cuModuleLoadData$' "$work/rt_$sm.log" || { echo "FAIL: plain sm_90 PTX was retargeted for an sm_$sm card"; fails=1; }
  done
  # a card that refuses the compiled module (the driver's JIT fails): the module stays in the simulator, the mode goes on, the output is the mode-off output
  rt90 rt_jit 86 STUB_CARD_OK=0
  [[ "$(grep -c '^cuModuleLoadData$' "$work/rt_jit.log")" == 1 ]] && grep -q 'cuModuleLoadData failed (1) for the card.s compute_90 PTX retargeted to sm_86' "$work/rt_jit.err" || { echo "FAIL: a failed JIT of a retargeted module was not said once or was retried: $(calls rt_jit); $(grep -m3 host_gpu "$work/rt_jit.err")"; fails=1; }
  ! grep -q 'mode off' "$work/rt_jit.err" || { echo "FAIL: a failed JIT ended turbo mode: $(grep -m1 'mode off' "$work/rt_jit.err")"; fails=1; }
  # 50. PTX with module variables is not given to the card.
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/varsp" vars > "$work/off_varsp.out" 2> "$work/off_varsp.err" || { echo "FAIL: the PTX build of host_gpu_vars with the mode off failed"; tail -5 "$work/off_varsp.err"; exit 1; }
  PROG=varsp run vp_ptx vars STUB_SM=90 STUB_MP=132 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the PTX-only vars run failed"; tail -5 "$work/vp_ptx.err"; fails=1; }
  ! grep -q '^cuModuleLoadData$' "$work/vp_ptx.log" || { echo "FAIL: PTX with module variables was given to the card: $(calls vp_ptx)"; fails=1; }
  grep -q 'the PTX declares a module variable' "$work/vp_ptx.err" || { echo "FAIL: the PTX's variables were not the reason: $(grep -m3 host_gpu "$work/vp_ptx.err")"; fails=1; }
  diff -q "$work/off_varsp.out" "$work/vp_ptx.out" > /dev/null || { echo "FAIL: turbo mode changes the output of the PTX build of host_gpu_vars"; fails=1; }
  # 51. a card that takes the module and refuses the launch (error 701): the mode ends, the simulator runs it, the output is the mode-off output.
  PROG=fit run lf_run wide VGPU_HOST_GPU=turbo STUB_CARD_OK=1 STUB_LAUNCH_FAIL=701 || { echo "FAIL: the run against a card that refuses the launch failed (exit $?): $(tail -3 "$work/lf_run.err")"; fails=1; }
  grep -q 'wide.* failed on the card (error 701); mode off, the simulator runs it' "$work/lf_run.err" || { echo "FAIL: a launch the card refused was not said: $(grep -m2 host_gpu "$work/lf_run.err")"; fails=1; }
  diff -q "$work/off_f_wide.out" "$work/lf_run.out" > /dev/null || { echo "FAIL: a launch the card refused changes the output of fit wide"; fails=1; }
  # 54. a module the card compiles from PTX obeys the turbo switches as a cubin does: a float atomic in it is kept out by VGPU_HOST_GPU_ATOMICS=0 and TURBO_OFF=atomics
  # (the class of a PTX module is that of all its kernels: its first kernel's launch is refused for the atomic of another)
  FIT_ATOM=1 PROG=fit75p run pa_on tiny VGPU_HOST_GPU=turbo || { echo "FAIL: the PTX atomic run failed"; fails=1; }
  grep -q '^cuModuleLoadData$' "$work/pa_on.log" && grep -q '^image ptx$' "$work/pa_on.log" || { echo "FAIL: PTX with a float atomic did not reach the card by default: $(calls pa_on)"; fails=1; }
  for sw in VGPU_HOST_GPU_ATOMICS=0 VGPU_HOST_GPU_TURBO_OFF=atomics; do
    FIT_ATOM=1 PROG=fit75p run pa_off tiny VGPU_HOST_GPU=turbo "$sw" || { echo "FAIL: the PTX atomic run with $sw failed"; fails=1; }
    ! grep -q '^cuModuleLoadData$' "$work/pa_off.log" || { echo "FAIL: $sw did not keep a PTX module with a float atomic out of the card: $(calls pa_off)"; fails=1; }
    grep -q '^host_gpu_refused: kernel=[^ ]* class=\(atomics\|off\) insn=PTX' "$work/pa_off.err" || { echo "FAIL: $sw: no class line for the PTX module: $(grep -m2 '^host_gpu_refused' "$work/pa_off.err")"; fails=1; }
  done
  # 55. variables in a card image that is not the simulator's: host_gpu_vars built for sm_80 and sm_86, an sm_80 card, the simulator on sm_86 (it runs the sm_86 image): the card's own
  # sm_80 cubin has the same layout, is given the simulator's addresses (no relocation left against a variable) and the kernels reach the card.
  PROG=vars80 run va_80 vars STUB_SM=80 STUB_MP=108 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the sm_80 card vars run failed"; tail -3 "$work/va_80.err"; fails=1; }
  grep -q '^image elf sm_80$' "$work/va_80.log" && grep -q '^relocs_to_variables 0$' "$work/va_80.log" || { echo "FAIL: the sm_80 cubin of a module with variables was not given to the sm_80 card with the simulator's addresses: $(calls va_80); $(grep -m3 host_gpu "$work/va_80.err")"; fails=1; }
  diff -q "$work/off_vvars.out" "$work/va_80.out" > /dev/null || { echo "FAIL: the sm_80 card changes the output of host_gpu_vars vars"; fails=1; }
  # 56. the driver starts while the program sets up (VGPU_HOST_GPU_FIT=1): a program that allocates and launches nothing has touched the stand-in driver by exit; with
  # VGPU_HOST_GPU_FIT=0, VGPU_HOST_GPU_PRESTART=0 or in exact mode it has not. The output is the mode-off output in each. The first launch of `tiny` finds the driver up.
  fitoff none
  PROG=fit run ps_on none VGPU_HOST_GPU=turbo || { echo "FAIL: the prestart run failed"; fails=1; }
  grep -q '^cuInit$' "$work/ps_on.log" || { echo "FAIL: turbo with FIT=1 did not start the driver while the program set up: $(calls ps_on)"; fails=1; }
  for sw in "VGPU_HOST_GPU=turbo VGPU_HOST_GPU_FIT=0" "VGPU_HOST_GPU=turbo VGPU_HOST_GPU_PRESTART=0" "VGPU_HOST_GPU=exact"; do
    PROG=fit run ps_off none $sw || { echo "FAIL: the run with $sw failed"; fails=1; }
    [[ ! -s "$work/ps_off.log" ]] || { echo "FAIL: $sw started the driver without a launch worth a card: $(calls ps_off)"; fails=1; }
    diff -q "$work/off_f_none.out" "$work/ps_off.out" > /dev/null || { echo "FAIL: $sw changes the output of fit none"; fails=1; }
  done
  diff -q "$work/off_f_none.out" "$work/ps_on.out" > /dev/null || { echo "FAIL: the prestart changes the output of fit none"; fails=1; }
  # ... a driver that cannot start (cuInit fails): said once, the mode is off, the output is the mode-off output, and no launch is held up
  PROG=fit run ps_bad tiny VGPU_HOST_GPU=turbo STUB_FAIL_INIT=1 || { echo "FAIL: the run with a driver that cannot start failed"; fails=1; }
  [[ "$(grep -c 'cuInit failed; mode off' "$work/ps_bad.err")" == 1 ]] || { echo "FAIL: a prestarted driver that fails was not said once: $(grep -m2 host_gpu "$work/ps_bad.err")"; fails=1; }
  ! grep -q '^cuModuleLoadData$' "$work/ps_bad.log" || { echo "FAIL: a launch reached the card after the prestarted driver failed"; fails=1; }
  diff -q "$work/off_f_tiny.out" "$work/ps_bad.out" > /dev/null || { echo "FAIL: a prestarted driver that fails changes the output"; fails=1; }
  # 57. what the SIMULATED profile and the kernel refuse is the simulator's to refuse, whatever the card would take (the stand-in card accepts any shape its limits allow): a kernel compiled
  # with __launch_bounds__(128) launched with 256 threads, which the simulator's launch validation refuses (launch-config), is refused by turbo mode the same way, with the same error, before
  # the card is asked, and the launch of 128 threads goes to the card; a shared-memory launch the T4 profile refuses (64 KiB a block at most) and the stand-in card (99 KiB) would take is refused
  # by the runtime API's check in both; and the reverse (an H100 profile, 227 KiB, on the card's 99 KiB: 120 KiB) is the simulator's to run, with the plain simulator's result.
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/fit" bounds > "$work/off_lb.out" 2> "$work/off_lb.err"; echo $? > "$work/off_lb.rc"
  grep -q 'launch=[1-9]' "$work/off_lb.out" && grep -q 'VirtualGPU error \[launch-config\]' "$work/off_lb.err" || { echo "FAIL: the plain simulator accepted 256 threads of a __launch_bounds__(128) kernel: $(cat "$work/off_lb.out") $(head -2 "$work/off_lb.err")"; fails=1; }
  PROG=fit run lb_t bounds VGPU_HOST_GPU=turbo STUB_CARD_OK=1 VGPU_HOST_GPU_LOG=1; echo $? > "$work/lb_t.rc"
  ! grep -q '^cuLaunchKernel$' "$work/lb_t.log" || { echo "FAIL: turbo mode sent a launch the simulator's validation refuses to the card: $(calls lb_t)"; fails=1; }
  diff -q "$work/off_lb.rc" "$work/lb_t.rc" > /dev/null && diff -q "$work/off_lb.out" "$work/lb_t.out" > /dev/null && diff <(grep 'VirtualGPU error' "$work/off_lb.err") <(grep 'VirtualGPU error' "$work/lb_t.err") > /dev/null || { echo "FAIL: turbo mode does not refuse the launch as the plain simulator does: rc $(cat "$work/off_lb.rc") against $(cat "$work/lb_t.rc"); $(grep 'VirtualGPU error' "$work/off_lb.err") against $(grep 'VirtualGPU error' "$work/lb_t.err")"; fails=1; }
  env FIT_THREADS=128 VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/fit" bounds > "$work/off_lb128.out" 2> "$work/off_lb128.err" || { echo "FAIL: the plain simulator refused 128 threads"; fails=1; }
  FIT_THREADS=128 PROG=fit run lb_ok bounds VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=1 STUB_CARD_OK=1 || { echo "FAIL: the turbo run of 128 threads failed"; fails=1; }
  grep -q '^cuLaunchKernel$' "$work/lb_ok.log" || { echo "FAIL: a launch inside the kernel's bounds did not reach the card: $(calls lb_ok)"; fails=1; }
  diff -q "$work/off_lb128.out" "$work/lb_ok.out" > /dev/null || { echo "FAIL: turbo mode changes the output of the 128-thread launch"; fails=1; }
  env FIT_SHARED_KB=30 VGPU_GPU=nvidia/t4 VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/fit75p" shared2 > "$work/off_t4.out" 2> "$work/off_t4.err"
  FIT_SHARED_KB=30 PROG=fit75p run sh_t4 shared2 VGPU_GPU=nvidia/t4 STUB_SM=75 STUB_MP=40 VGPU_HOST_GPU=turbo STUB_CARD_OK=1 VGPU_HOST_GPU_LOG=1
  grep -q 'launch=[1-9]' "$work/off_t4.out" && ! grep -q '^cuLaunchKernel$' "$work/sh_t4.log" && diff -q "$work/off_t4.out" "$work/sh_t4.out" > /dev/null && diff <(grep 'VirtualGPU error' "$work/off_t4.err") <(grep 'VirtualGPU error' "$work/sh_t4.err") > /dev/null || { echo "FAIL: 70 KiB of shared memory on a T4 profile is not refused alike by the plain simulator and turbo mode: $(cat "$work/off_t4.out" "$work/sh_t4.out" | tr '\n' ' ')"; fails=1; }
  env FIT_SHARED_KB=120 VGPU_GPU=nvidia/h100 VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/fit" shared > "$work/off_h120.out" 2> "$work/off_h120.err" || { echo "FAIL: the plain simulator refused 120 KiB on an H100 profile: $(head -2 "$work/off_h120.err")"; fails=1; }
  FIT_SHARED_KB=120 PROG=fit run sh_h120 shared VGPU_GPU=nvidia/h100 STUB_SM=86 STUB_MP=28 VGPU_HOST_GPU=turbo STUB_CARD_OK=1 VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the turbo run of 120 KiB on an H100 profile failed"; fails=1; }
  ! grep -q '^cuLaunchKernel$' "$work/sh_h120.log" && diff -q "$work/off_h120.out" "$work/sh_h120.out" > /dev/null || { echo "FAIL: 120 KiB on an H100 profile (over the card's 99 KiB) was not the simulator's: $(calls sh_h120)"; fails=1; }
  # 52. the notice.
  grep -q '^host_gpu: VGPU_HOST_GPU=turbo runs the kernels it can on the host GPU.*results may differ from the simulator' "$work/fit_tiny.err" || { echo "FAIL: turbo mode did not say that its results may differ: $(grep -m2 host_gpu "$work/fit_tiny.err")"; fails=1; }
  [[ "$(grep -c 'may differ from the simulator' "$work/fit_tiny.err")" == 1 ]] || { echo "FAIL: the notice was not said exactly once"; fails=1; }
  grep -q 'may differ from the simulator' "$work/fit_exact.err" && { echo "FAIL: exact mode said that its results may differ"; fails=1; }
  PROG=fit run fit_quiet tiny VGPU_QUIET=1 VGPU_HOST_GPU=turbo || { echo "FAIL: the quiet turbo run failed"; fails=1; }
  grep -q 'may differ from the simulator' "$work/fit_quiet.err" && { echo "FAIL: VGPU_QUIET=1 did not silence the notice"; fails=1; }
  # 53. WSL2: turbo with FIT=1 is off at once (the driver is never touched); FIT=0 and exact are not asked.
  echo "Linux version 6.18.0-microsoft-standard-WSL2" > "$work/wslver"
  PROG=fit run wsl_fit tiny VGPU_HOST_GPU=turbo VGPU_HOST_GPU_PROC_VERSION="$work/wslver" || { echo "FAIL: the turbo run on a WSL host failed"; fails=1; }
  [[ ! -s "$work/wsl_fit.log" ]] && grep -q 'this host is WSL2.*mode off' "$work/wsl_fit.err" || { echo "FAIL: turbo with FIT=1 on a WSL host was not turned off at once: $(calls wsl_fit); $(grep -m2 host_gpu "$work/wsl_fit.err")"; fails=1; }
  diff -q "$work/off_f_tiny.out" "$work/wsl_fit.out" > /dev/null || { echo "FAIL: the WSL notice changes the output"; fails=1; }
  PROG=fit run wsl_rules tiny VGPU_HOST_GPU=turbo VGPU_HOST_GPU_FIT=0 VGPU_HOST_GPU_PROC_VERSION="$work/wslver" "${reachx[@]}" || { echo "FAIL: the FIT=0 run on a WSL host failed"; fails=1; }
  ! grep -q 'this host is WSL2' "$work/wsl_rules.err" || { echo "FAIL: FIT=0 was turned off for WSL"; fails=1; }
  PROG=fit run wsl_exact tiny VGPU_HOST_GPU=exact VGPU_HOST_GPU_PROC_VERSION="$work/wslver" || { echo "FAIL: the exact run failed"; fails=1; }
  ! grep -q 'this host is WSL2' "$work/wsl_exact.err" || { echo "FAIL: exact mode was turned off for WSL"; fails=1; }
else
  echo "NOTE: turbo fit steps 45 to 52 need the rtx3060 profile (sm_86); skipped for $profile"
fi
# 58. (#660, #662) what a kernel kept in the simulator costs the mode: 50 kernels of one launch each and a looping kernel stay in the simulator and say so. These pin the size rules, which
# turbo mode asks with VGPU_HOST_GPU_FIT=0 (the default FIT=1 would send every one of them to the card).
for arg in kernels repeat; do
  env VGPU_GPU="$profile" VGPU_THREADS=4 LD_LIBRARY_PATH="$shim" "$work/ovh" "$arg" > "$work/off_ovh_$arg.out" 2> /dev/null || { echo "FAIL: the $arg run with the mode off failed"; fails=1; }
  for mode in exact turbo; do
    PROG=ovh run "ovh_${mode}_$arg" "$arg" VGPU_HOST_GPU=$mode VGPU_HOST_GPU_FIT=0 VGPU_HOST_GPU_LOG=1 || { echo "FAIL: the $mode $arg run failed"; fails=1; }
    [[ ! -s "$work/ovh_${mode}_$arg.log" ]] || { echo "FAIL: $mode $arg reached the driver: $(calls "ovh_${mode}_$arg")"; fails=1; }
    diff -q "$work/off_ovh_$arg.out" "$work/ovh_${mode}_$arg.out" > /dev/null || { echo "FAIL: $mode changes the output of $arg"; fails=1; }
    grep -q '^host_gpu: 0 launches ran on the card, ' "$work/ovh_${mode}_$arg.err" || { echo "FAIL: $mode $arg: a launch went to the card: $(grep '^host_gpu: ' "$work/ovh_${mode}_$arg.err" | head -1)"; fails=1; }
  done
done
n="$(grep -c '^host_gpu_overhead: kept_launches=50 .* kernels=50$' "$work/ovh_turbo_kernels.err")"
(( n == 1 )) || { echo "FAIL: turbo kernels: expected 50 kept launches of 50 kernels: $(grep '^host_gpu_overhead' "$work/ovh_turbo_kernels.err")"; fails=1; }
n="$(grep -c '^host_gpu: _Z3k[0-9]*Pf: card image [^;]*; classes: exact; admitted' "$work/ovh_turbo_kernels.err")"
(( n == 50 )) || { echo "FAIL: turbo kernels: expected 50 kernels classed exact and admitted, got $n"; fails=1; }
n="$(grep -c '^host_gpu_nosample: kernel=_Z3k[0-9]*Pf .* loop=0 ' "$work/ovh_turbo_kernels.err")"
(( n == 50 )) || { echo "FAIL: turbo kernels: expected 50 kernels without a loop, got $n"; fails=1; }
run ovh_loop acct VGPU_HOST_GPU=turbo VGPU_HOST_GPU_FIT=0 VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_START_COST=1000 || { echo "FAIL: the turbo acct run failed"; fails=1; }
grep -q '^host_gpu_nosample: .* loop=1 ' "$work/ovh_loop.err" || { echo "FAIL: acct, whose kernel loops, was not seen to loop: $(grep -m2 'host_gpu_nosample' "$work/ovh_loop.err")"; fails=1; }
(( fails == 0 )) && { echo "PASS: first launch $first ms against a driver start of $init_ms ms; driver calls in the big run: $(calls on_big); many launches of ${many_us:-?} us started the driver after ${many_at:-?} of them"; exit 0; }
exit 1
