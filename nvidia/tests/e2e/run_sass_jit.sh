#!/usr/bin/env bash
# Tier 2 of the SASS executor (src/sass/jit.hpp, VGPU_JIT=1), milestone 1, against the tier (VGPU_TIER=1).
#
# The programs are the ones run_sass_tier.sh builds (sass_tier for every architecture, sass_tier_steps and the other SASS test
# programs): this script has that script build them (VGPU_TIER_BUILD_ONLY, VGPU_TIER_WORK) and then runs each twice on its generation's
# simulated GPU, one host thread, the performance counters on: with the tier, and with the JIT forced on at once (VGPU_JIT_THRESHOLD=1,
# synchronous compile, so that the first region a warp meets is compiled and every later one runs as generated code). The two must pass
# and print byte for byte the same: every result and every counter of every launch, the text of any error, and the tier's own statistics
# of what ran natively (VGPU_TIER_STATS: the JIT calls the tier's handlers, so these counts are the tier's too).
#
# More on sass_tier: four host threads with the JIT; the JIT again with the cache the first run filled (nothing compiles, same output);
# the background compile (no synchronous mode: regions are used when they are ready, whenever that is); the chain off.
# jit_loops (loops and branches: a uniform loop the generated code runs on its own, one every lane leaves at its own iteration, nested loops with an early exit, a store in
# the body, a body longer than a region, the clock around a loop, barriers, guarded ops, uniform values that steer a loop) gets sass_tier's variants.
# On sass_tier_steps (the store gate, the step limit, the 256-instruction run, faults deep in straight code): every fault program,
# tier against JIT, with and without -fexceptions for the generated code.
# And the failure modes (no compiler, a compiler that fails, a cache that cannot be written): output identical, the JIT never ran.
#
# Two sizes. The default, run on every pull request (ctest e2e_sass_jit), is the gate: sass_tier and sass_tier_steps on sm_86 with every variant above,
# one random-kernel seed, and the failure modes; only those two programs are built. `--full` (or VGPU_JIT_FULL=1; ctest e2e_sass_jit_full) is the whole
# matrix: every program on every architecture, three random seeds. It costs an hour on an idle 16-CPU machine, so it SKIPs unless VGPU_NIGHTLY=1, which
# the schedule-only `jit` job of sass-ptx.yml sets.
# VGPU_JIT_PROGRAMS="sass_tier.sm_86 sass_tier_steps.sm_86" narrows the programs (and drops the random stage); VGPU_TIER_WORK keeps the built programs between runs.
#
# VGPU_JIT_SHARD=tier,steps,steps_gather,steps_shared,random,modes runs only those stages of the gate (default: all of them; "programs" is every other program of --full), so
# that ctest can run them side by side: e2e_sass_jit_shard_tier, _steps, _steps_gather, _steps_shared, _random and _modes in CMakeLists.txt together are exactly the stages of the
# whole run, once each. VGPU_JIT_LIST=1 prints the stages a run (with its VGPU_JIT_SHARD) would run, one a line, without building or running anything;
# tests/lint/check_shard_partition.sh uses it to prove the shards partition the whole.
#
# The comparison cannot pass by the JIT doing nothing: sass_tier's statistics must show the generated regions ran at least
# VGPU_JIT_MIN_PERCENT (default 10) percent of its warp instructions.
set -uo pipefail
shard="${VGPU_JIT_SHARD:-}"
# Is stage $1 (tier, loops, steps, steps_gather, steps_shared, programs, random, modes) in this run's shard?
want() { [[ -z $shard || ",$shard," == *",$1,"* ]]; }
if [[ -n ${VGPU_JIT_LIST:-} ]]; then
  if [[ ${1:-} == --full || -n ${VGPU_JIT_FULL:-} ]]; then stages="tier loops steps steps_gather steps_shared programs random modes"; else stages="tier loops steps steps_gather steps_shared random modes"; fi
  for st in $stages; do want "$st" && echo "$st"; done
  exit 0
fi
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
if [[ -n $(shim_sanitizer "$shim") ]]; then echo "SKIP: the JIT is off in sanitizer builds"; exit 0; fi
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo "SKIP: no C++ compiler on PATH"; exit 0; }
if [[ ${1:-} == --full || -n ${VGPU_JIT_FULL:-} ]] && [[ -z ${VGPU_NIGHTLY:-} ]]; then
  echo "SKIP: the full JIT matrix runs in the nightly (set VGPU_NIGHTLY=1); ctest e2e_sass_jit is the gate"; exit 0
fi
if [[ -n ${VGPU_TIER_WORK:-} ]]; then
  work="$VGPU_TIER_WORK"; mkdir -p "$work"
else
  work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-jit.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
fi
export VGPU_TIER_WORK="$work"
full=0
[[ ${1:-} == --full || -n ${VGPU_JIT_FULL:-} ]] && full=1
prog_filter="${VGPU_JIT_PROGRAMS:-}"
if (( ! full )) && [[ -z $prog_filter ]]; then   # the gate: two programs, built for one architecture
  prog_filter="sass_tier.sm_86 sass_tier_steps.sm_86 jit_loops.sm_86"
  export VGPU_TIER_ARCHS="${VGPU_TIER_ARCHS:-sm_86}" VGPU_TIER_PROGRAMS="${VGPU_TIER_PROGRAMS:-sass_tier_steps:75 jit_loops:75}"
fi
if [[ -n $shard ]]; then   # build only what this shard's stages run (sass_tier is also the failure modes' program)
  want tier || want modes || export VGPU_TIER_NO_SASS_TIER=1
  if ! { want steps || want steps_gather || want steps_shared || want programs; }; then   # (only the loops shard: that program alone)
    want loops && [[ -z ${VGPU_TIER_PROGRAMS:-} || $VGPU_TIER_PROGRAMS == *jit_loops* ]] && export VGPU_TIER_PROGRAMS=jit_loops:75 || export VGPU_TIER_PROGRAMS=-
  elif ! want loops && [[ ${VGPU_TIER_PROGRAMS:-} == *jit_loops* ]]; then   # (the steps shards without the loops program)
    export VGPU_TIER_PROGRAMS="${VGPU_TIER_PROGRAMS//jit_loops:75/}"
  fi
  if [[ -n ${VGPU_TIER_NO_SASS_TIER:-} && ${VGPU_TIER_PROGRAMS:-} == - ]]; then built_nothing=1; fi
fi
if [[ -n ${built_nothing:-} ]]; then rc=0; build_log=""; else
build_log="$(VGPU_TIER_BUILD_ONLY=1 "$root/nvidia/tests/e2e/run_sass_tier.sh" 2>&1)"; rc=$?
if grep -q '^SKIP' <<< "$build_log" && ! compgen -G "$work/*.sm_*" >/dev/null; then echo "$build_log" | grep '^SKIP' | head -1; exit 0; fi
(( rc == 0 )) || { echo "FAIL: building the test programs"; echo "$build_log" | tail -20; exit 1; }
fi

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4 [sm_90]=nvidia/h100
                [sm_100]=nvidia/b200 [sm_120]=nvidia/rtx5090)
min_percent="${VGPU_JIT_MIN_PERCENT:-10}"
run_limit="${VGPU_TIER_TIMEOUT:-900}"
cache="$work/jit-cache"
rm -rf "$cache"
fails=0
fault_throws=""
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
san="${san_flags[*]}"

# Runs $bin with the shim, one host thread, counters on, quiet; $1 is the label of the extra environment that follows.
runit() {
  local bin=$1 arch=$2; shift 2
  (cd "$work" && env VGPU_QUIET=1 VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") "$@" timeout "$run_limit" "$bin" 2>&1)
}
jitenv=(VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2 "VGPU_JIT_CACHE=$cache")

# A JIT run of one of sass_tier_steps' fault programs (or of the step limit), whose whole output, stderr and all, is compared with the
# tier's: jit_fault <cache> [VAR=value ...] -- <program arguments>. The comparison means something only for a run in which the JIT did
# compile. On a loaded machine a compile can outlast its time limit (120 s by default; the compiler runs at a lower priority) or the
# compiler be killed; the regions then stay on the tier, which reports the same fault, and the JIT says so on stderr ("pantheonsim: JIT:
# ... stay on the tier"), which was then the only difference: "the scatter fault is reported differently with the JIT", now and then, on
# the busiest runner (#611; 72 runs of tier, JIT and JIT without exceptions on an idle machine and under load print the same, byte for
# byte). So these runs give a compile 840 s, and a run that still could not compile is taken once more, with a cache of its own; if that
# one cannot either, its output is what is compared, and the failure names the notice.
jit_fault() {
  local c=$1 out try
  shift
  local -a envs=()
  while (( $# )) && [[ $1 != -- ]]; do envs+=("$1"); shift; done
  shift
  for try in 1 2; do
    out="$(cd "$work" && env "${fenv[@]}" "${jitenv[@]}" VGPU_JIT_TIMEOUT=840 "VGPU_JIT_CACHE=$c" "${envs[@]}" timeout 900 "$bin" "$@" 2>&1)"
    grep -q '^pantheonsim: JIT: ' <<< "$out" || break
    c="$c-again"; rm -rf "$c"
  done
  printf '%s\n' "$out"
}
# Why a fault run of the JIT ($2) differs from the tier's ($1), for the message: the JIT's notice if it could not compile, else the diff.
jit_fault_diff() {
  local n
  n="$(grep -m1 '^pantheonsim: JIT: ' <<< "$2")"
  if [[ -n $n ]]; then echo "  (it could not compile: ${n#pantheonsim: JIT: })"; else echo "  ($(diff <(echo "$1") <(echo "$2") | grep '^[<>]' | head -4 | tr '\n' ' '))"; fi
}

# The "jit: ..." line of a statistics file, and one number from it.
jit_line() { grep -h '^jit: ' "$1" 2>/dev/null | tail -1; }
jit_num() { sed -n "s/.*[ ,;]\([0-9][0-9]*\) $2.*/\1/p" <<< "$1" | tail -1; }

check() {
  local p=$1 arch=$2 bin="$work/$1.$2" a b ra rb why="" sa sb st
  [[ -x $bin ]] || return
  sa="$work/$p.$arch.tier.stats"; sb="$work/$p.$arch.jit.stats"; st="$work/$p.$arch.jit2.stats"
  rm -f "$sa" "$sb" "$st"
  a="$(runit "$bin" $arch VGPU_TIER=1 VGPU_TIER_STATS=1 VGPU_TIER_STATS_FILE="$sa")"; ra=$?
  b="$(runit "$bin" $arch "${jitenv[@]}" VGPU_TIER_STATS=1 VGPU_TIER_STATS_FILE="$sb" VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$sb")"; rb=$?
  grep -q "running SASS" <<< "$a" || why="its SASS did not run"
  [[ $ra == "$rb" ]] || why="${why:+$why; }exit status $ra with the tier, $rb with the JIT"
  [[ "$a" == "$b" ]] || why="${why:+$why; }the tier and the JIT print differently"
  # The tier's statistics: what ran natively, per mnemonic (the JIT's handlers are the tier's, so the counts must be the same).
  local na nb
  na="$(grep -h '^tier native:\|^tier: ' "$sa" 2>/dev/null)"; nb="$(grep -h '^tier native:\|^tier: ' "$sb" 2>/dev/null)"
  [[ "$na" == "$nb" ]] || why="${why:+$why; }the tier's statistics differ with the JIT on"
  if [[ $ra != 0 && $p != sass_tier && $p != jit_loops && $ra == "$rb" && "$a" == "$b" ]]; then
    echo "ok   $arch $p  (fails identically in both, exit $ra: not compared further)"
    return
  fi
  [[ $ra == 0 ]] || why="${why:+$why; }the run failed (exit $ra)"
  local jl regions
  jl="$(jit_line "$sb")"
  regions="$(jit_num "$jl" 'region calls')"
  # Four host threads: with the JIT, what one thread of the tier prints -- for a program whose output does not depend on the number of
  # threads (the tier's own four-thread run prints what its one-thread run does; sass_tier is required to be one).
  if [[ -z $why ]]; then
    local t4 c4
    t4="$(runit "$bin" $arch VGPU_TIER=1 VGPU_THREADS=4)"
    if [[ "$t4" == "$a" ]]; then
      c4="$(runit "$bin" $arch "${jitenv[@]}" VGPU_THREADS=4)"
      [[ "$a" == "$c4" ]] || why="${why:+$why; }four threads with the JIT print differently from the tier's"
    elif [[ $p == sass_tier || $p == jit_loops ]]; then
      why="${why:+$why; }the tier's four threads print differently from its one ($p must not)"
    fi
  fi
  if [[ $p == sass_tier || $p == jit_loops ]]; then
    local ran total
    ran="$(sed -n 's/.*ran \([0-9]*\) warp instructions.*/\1/p' <<< "$jl")"
    total="$(sed -n 's/^tier: \([0-9]*\) of \([0-9]*\) warp.*/\2/p' "$sb" | head -1)"
    if [[ -z $ran || -z $total ]] || (( total == 0 || ran * 100 < total * min_percent )); then
      why="${why:+$why; }the JIT ran ${ran:-?} of ${total:-?} warp instructions (wanted at least $min_percent%)"
    fi
    # Again, with the cache the first run filled: nothing compiles, and the output is the same.
    local c2 jl2
    c2="$(runit "$bin" $arch "${jitenv[@]}" VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$st")"
    jl2="$(jit_line "$st")"
    # The loops program must have had loops run in generated code: branches and register-only ops that the generated code went on past (not with the tier's
    # statistics on, which keep the run loop from going on past them: the first run above has them on, this one has not).
    if [[ $p == jit_loops ]]; then
      local cont
      cont="$(sed -n 's/.*[ ,;]\([0-9][0-9]*\) branches and register-only ops.*/\1/p' <<< "$jl2" | tail -1)"
      (( ${cont:-0} >= ${VGPU_JIT_MIN_CONTINUED:-1000} )) || why="${why:+$why; }only ${cont:-0} branches went on inside generated code"
    fi
    [[ "$a" == "$c2" ]] || why="${why:+$why; }the JIT with a warm cache prints differently from the tier"
    [[ "$(jit_num "$jl2" 'compiles')" == 0 ]] || why="${why:+$why; }a run with a warm cache compiled again ($jl2)"
    # The background compile: whenever it finishes, the output is the same.
    local bg
    bg="$(runit "$bin" $arch VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 "VGPU_JIT_CACHE=$work/jit-cache-bg")"
    [[ "$a" == "$bg" ]] || why="${why:+$why; }the JIT with the compile in the background prints differently from the tier"
    # No chain.
    local nc
    nc="$(runit "$bin" $arch "${jitenv[@]}" VGPU_TIER_CHAIN=0)"
    [[ "$a" == "$nc" ]] || why="${why:+$why; }the JIT without the tier's chain prints differently from the tier"
    # The share rule on (the settings above compile everything: HOT_PERMILLE=0 turns it off): a sample of 5000 warp instructions, and the bar at 1%
    # (nearly every section kept), 60% (the default) and 100% (nearly every section declined and back on the plain loop). The output is the tier's in all.
    local sh
    for bar in 1 60 100; do
      sh="$(runit "$bin" $arch VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=20000 VGPU_JIT_SYNC=1 VGPU_JIT_SAMPLE=5000 VGPU_JIT_MIN_SHARE=$bar VGPU_JIT_MIN_REGION=2 "VGPU_JIT_CACHE=$work/jit-cache-share$bar")"
      [[ "$a" == "$sh" ]] || why="${why:+$why; }the JIT with the share rule at $bar% prints differently from the tier"
    done
  fi
  if [[ $p == sass_tier_steps && -z $why ]]; then
    # Faults deep into a run of straight instructions, and the step limit, with the generated code compiled with and without -fexceptions.
    local f fa fb fc thr=0 fsf fenv=(VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_MAX_STEPS=50000 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim"))
    if want steps; then
    for f in gather scatter local shared atomic trap forever; do
      fa="$(cd "$work" && env "${fenv[@]}" VGPU_TIER=1 timeout 300 "$bin" fault $f 2>&1)"
      fsf="$work/fault.$f.stats"; rm -f "$fsf"
      fb="$(jit_fault "$cache" VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$fsf" -- fault $f)"
      thr=$(( thr + $(sed -n 's/.* \([0-9][0-9]*\) exceptions unwound.*/\1/p' "$fsf" 2>/dev/null | tail -1 | grep . || echo 0) ))
      fc="$(jit_fault "$cache-noexc" VGPU_JIT_FLAGS=-fno-exceptions -- fault $f)"
      if ! grep -q "^fault $f: " <<< "$fa" || grep -q "^fault $f: no error" <<< "$fa"; then why="${why:+$why; }the $f fault did not happen"
      elif [[ "$fa" != "$fb" ]]; then why="${why:+$why; }the $f fault is reported differently with the JIT$(jit_fault_diff "$fa" "$fb")"
      elif [[ "$fa" != "$fc" ]]; then why="${why:+$why; }the $f fault is reported differently with the JIT built without exceptions$(jit_fault_diff "$fa" "$fc")"; fi
    done
    # The retry of jit_fault, checked (#611): with a compiler that is killed on its first compile (a stand-in for one that outlasts its
    # limit on a loaded machine), the scatter fault is still reported as the tier reports it.
    local flaky="$work/flaky-c++" real_cxx fl ft
    real_cxx="$(command -v c++ || command -v g++ || command -v clang++)"
    printf '#!/bin/sh\ncase " $* " in *" -shared "*) mkdir "%s.once" 2>/dev/null && kill -9 $$ ;; esac\nexec "%s" "$@"\n' "$flaky" "$real_cxx" > "$flaky"
    chmod +x "$flaky"; rm -rf "$flaky.once" "$cache-flaky" "$cache-flaky-again"
    ft="$(cd "$work" && env "${fenv[@]}" VGPU_TIER=1 timeout 300 "$bin" fault scatter 2>&1)"
    fl="$(jit_fault "$cache-flaky" VGPU_JIT_CXX="$flaky" -- fault scatter)"
    if [[ ! -d $flaky.once ]]; then why="${why:+$why; }the compiler that fails once was never asked to compile"
    elif [[ "$ft" != "$fl" ]]; then why="${why:+$why; }after one failed compile the scatter fault is reported differently with the JIT$(jit_fault_diff "$ft" "$fl")"; fi
    # The step budget is checked once for a whole region (steps + k): a limit that falls inside the region the fault is in must give the
    # limit's error, not the fault's. For each fault that is thrown inside a region: the smallest VGPU_MAX_STEPS at which the tier reports the
    # fault (found by bisection), then the limits just below it, tier against JIT.
    fi
    local lim la lb lo hi mid off
    for f in gather shared; do
      want steps_$f || continue
      lo=1; hi=20000
      while (( lo < hi )); do
        mid=$(( (lo + hi) / 2 ))
        la="$(cd "$work" && env "${fenv[@]}" VGPU_MAX_STEPS=$mid VGPU_TIER=1 timeout 300 "$bin" fault $f 2>&1)"
        if grep -q "illegal memory access" <<< "$la"; then hi=$mid; else lo=$((mid + 1)); fi
      done
      for off in 0 1 2 3 5 8 13 21 34 55 89 97; do
        lim=$(( lo - off ))
        (( lim >= 1 )) || continue
        la="$(cd "$work" && env "${fenv[@]}" VGPU_MAX_STEPS=$lim VGPU_TIER=1 timeout 300 "$bin" fault $f 2>&1)"
        lb="$(jit_fault "$cache" VGPU_MAX_STEPS=$lim -- fault $f)"
        [[ "$la" == "$lb" ]] || why="${why:+$why; }$f with VGPU_MAX_STEPS=$lim (the fault is at step $lo): the JIT reports '$(head -c 80 <<< "$lb")' where the tier reports '$(head -c 80 <<< "$la")'"
      done
    done
    # The most a region can hold: the chunk of 256 straight instructions the run loop gives a turn is checked once for a whole region, and a
    # region of 257 ops (the 256-instruction run, then the one that ends it) must be left to the tier. Output of the whole program, all
    # three limits against the tier's.
    local cap co
    if want steps; then
    for cap in 255 256 257; do
      co="$(runit "$bin" $arch "${jitenv[@]}" VGPU_JIT_MAX_REGION=$cap "VGPU_JIT_CACHE=$cache-cap$cap")"
      [[ "$a" == "$co" ]] || why="${why:+$why; }regions of up to $cap ops print differently from the tier"
    done
    # The comparison above means something only if faults really were thrown from inside generated regions.
    (( thr >= ${VGPU_JIT_MIN_THROWS:-2} )) || why="${why:+$why; }only $thr of the faults were thrown inside a generated region"
    fault_throws=$thr
    fi
  fi
  if [[ -n $why ]]; then
    echo "FAIL $arch $p: $why"
    diff <(echo "$a") <(echo "$b") | head -12 | sed 's/^/    /'
    fails=1
  else
    echo "ok   $arch $p  (${jl#jit: })${fault_throws:+  [$fault_throws faults thrown inside regions]}"
    fault_throws=""
  fi
}

shopt -s nullglob
programs=()
for f in "$work"/*.sm_*; do
  [[ -x $f && $f != *.build && $f != *.stats ]] || continue
  b="${f##*/}"
  # VGPU_JIT_PROGRAMS="name.arch name.arch" narrows the list (the mutation checks run two programs, not fifty).
  [[ -z $prog_filter || " $prog_filter " == *" $b "* ]] || continue
  programs+=("${b%.*} ${b##*.}")
done
shopt -u nullglob
(( ${#programs[@]} )) || [[ -n ${built_nothing:-} ]] || { echo "SKIP: no test programs were built"; exit 0; }
if [[ -z ${VGPU_JIT_ONLY_RANDOM:-} ]]; then
  for j in "${programs[@]}"; do
    case ${j%% *} in
      sass_tier) want tier && check $j ;;
      jit_loops) want loops && check $j ;;
      sass_tier_steps) { want steps || want steps_gather || want steps_shared; } && check $j ;;   # check runs only the sections of the stages it has
      *) want programs && check $j ;;
    esac
  done
fi

# Random straight-line kernels (jit_random.py): u32 / u64 / float statements over a small pool of variables, guards, loads; blocks of 128, 48 and
# 33 threads. The tier and the JIT with every region compiled must print the same hashes and counters (VGPU_JIT_RANDOM_SEEDS seeds of
# VGPU_JIT_RANDOM_KERNELS kernels each; default 3 of 16 with --full, 1 in the gate; VGPU_JIT_RANDOM_ARCH, default sm_86).
if want random && command -v python3 >/dev/null 2>&1 && [[ -z ${VGPU_JIT_PROGRAMS:-} ]]; then
  rarch="${VGPU_JIT_RANDOM_ARCH:-sm_86}"
  # (A rig without nvcc runs the programs built for it beforehand, jit_random.<seed>.<arch> in the work directory.)
  if [[ -z $nvcc_bin ]] && compgen -G "$work/jit_random.*.$rarch" >/dev/null || grep -qx "$rarch" <<< "$("$nvcc_bin" --list-gpu-code 2>/dev/null)"; then
    for seed in $(seq 1 "${VGPU_JIT_RANDOM_SEEDS:-$(( full ? 3 : 1 ))}"); do
      src="$work/jit_random.$seed.cu"; bin="$work/jit_random.$seed.$rarch"
      if [[ -n $nvcc_bin ]]; then
        python3 "$root/nvidia/tests/e2e/jit_random.py" "$seed" "${VGPU_JIT_RANDOM_KERNELS:-16}" > "$src"
        if ! "$nvcc_bin" -std=c++17 -arch="$rarch" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san "$src" -o "$bin" -lcuda 2> "$bin.build"; then
          echo "FAIL $rarch jit_random seed $seed: does not build"; tail -3 "$bin.build" | sed 's/^/    /'; fails=1; continue
        fi
      elif [[ ! -x $bin ]]; then
        continue
      fi
      ra="$(runit "$bin" $rarch VGPU_TIER=1)"; rb="$(runit "$bin" $rarch "${jitenv[@]}" VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$work/jr.$seed.stats")"
      rb="$(runit "$bin" $rarch "${jitenv[@]}")"; rc="$(runit "$bin" $rarch "${jitenv[@]}" VGPU_THREADS=4)"
      jl="$(jit_line "$work/jr.$seed.stats")"
      if [[ "$ra" != "$rb" ]]; then echo "FAIL $rarch jit_random seed $seed: the tier and the JIT print differently"; diff <(echo "$ra") <(echo "$rb") | head -6 | sed 's/^/    /'; fails=1
      elif [[ "$(runit "$bin" $rarch VGPU_TIER=1 VGPU_THREADS=4)" == "$ra" && "$ra" != "$rc" ]]; then echo "FAIL $rarch jit_random seed $seed: four threads with the JIT print differently"; fails=1
      else echo "ok   $rarch jit_random seed $seed  (${jl#jit: })"; fi
    done
  fi
fi

# The failure modes: the JIT cannot do its job and the program must not notice (stdout and stderr, the notices aside: VGPU_QUIET).
prog=sass_tier; arch=sm_86
if want modes && [[ -z ${VGPU_JIT_ONLY_RANDOM:-} && -x "$work/$prog.$arch" ]]; then
  bin="$work/$prog.$arch"
  ref="$(runit "$bin" $arch VGPU_TIER=1)"
  mode() {
    local label=$1; shift
    local out sf="$work/mode.stats"
    rm -f "$sf"
    out="$(runit "$bin" $arch VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$sf" "$@")"
    if [[ "$out" != "$ref" ]]; then echo "FAIL $arch $prog ($label): output differs from the tier"; diff <(echo "$ref") <(echo "$out") | head -6 | sed 's/^/    /'; fails=1; return; fi
    local line n
    line="$(jit_line "$sf")"; n="$(jit_num "$line" 'region calls')"
    if [[ -n $n && $n != 0 ]]; then echo "FAIL $arch $prog ($label): generated regions ran although the JIT could not work ($line)"; fails=1; return; fi
    echo "ok   $arch $prog ($label)  (${line#jit: })"
  }
  mode "no compiler" VGPU_JIT_CXX=/nonexistent/c++ "VGPU_JIT_CACHE=$work/jit-c1"
  mode "a compiler that fails" "VGPU_JIT_FLAGS=-this-flag-does-not-exist" "VGPU_JIT_CACHE=$work/jit-c2"
  mode "a cache that cannot be created" "VGPU_JIT_CACHE=/proc/no/such/dir"
  mkdir -p "$work/jit-c3" && chmod 0777 "$work/jit-c3"
  mode "a cache others can write to" "VGPU_JIT_CACHE=$work/jit-c3"
  fakecc="$work/slowcc"
  printf '#!/bin/sh\nif [ "$1" = --version ]; then echo slowcc; exit 0; fi\nif [ "$1" = -dumpmachine ]; then echo m; exit 0; fi\nsleep 30\n' > "$fakecc"; chmod +x "$fakecc"
  mode "a compile that times out" "VGPU_JIT_CXX=$fakecc" VGPU_JIT_TIMEOUT=1 "VGPU_JIT_CACHE=$work/jit-c4"

  # The forms of the generated code (round sp38): the runs above and in the other stages are the default, the cheap form (rows in the register
  # file, one body per region); the full form (VGPU_JIT_RESIDENCE=1 VGPU_JIT_SPECIALIZE=1: rows in locals, a second body for whole warps) is
  # still a knob, and must print what the tier prints with regions running, as each part of it alone must.
  form() {
    local label=$1; shift
    local out sf="$work/form.stats" line n
    rm -f "$sf"
    out="$(runit "$bin" $arch VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2 VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$sf" "$@")"
    if [[ "$out" != "$ref" ]]; then echo "FAIL $arch $prog ($label): output differs from the tier"; diff <(echo "$ref") <(echo "$out") | head -6 | sed 's/^/    /'; fails=1; return; fi
    line="$(jit_line "$sf")"; n="$(jit_num "$line" 'region calls')"
    if [[ -z $n || $n == 0 ]]; then echo "FAIL $arch $prog ($label): no generated region ran ($line)"; fails=1; return; fi
    echo "ok   $arch $prog ($label)  (${line#jit: })"
  }
  form "the full form: residence and specialisation" VGPU_JIT_RESIDENCE=1 VGPU_JIT_SPECIALIZE=1 "VGPU_JIT_CACHE=$work/jit-f1"
  form "residence only" VGPU_JIT_RESIDENCE=1 "VGPU_JIT_CACHE=$work/jit-f2"
  form "specialisation only" VGPU_JIT_SPECIALIZE=1 "VGPU_JIT_CACHE=$work/jit-f3"
  # Linking (round sp52, VGPU_JIT_LINK=1): the generated code goes on past branches, register-only ops, BSSY, EXITs no lane takes, CALL.REL and RET.REL
  # where the run loop's chain would pick its group again, runs branches that split the group, and the selected regions of a section are one function.
  # The same output as the tier with regions running, in the cheap form and in the full one.
  form "linking" VGPU_JIT_LINK=1 "VGPU_JIT_CACHE=$work/jit-l1"
  form "linking, the full form" VGPU_JIT_LINK=1 VGPU_JIT_RESIDENCE=1 VGPU_JIT_SPECIALIZE=1 "VGPU_JIT_CACHE=$work/jit-l2"
fi

[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: the tier and the JIT differ"
exit $fails
