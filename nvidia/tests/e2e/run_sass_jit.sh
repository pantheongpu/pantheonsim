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
# On sass_tier_steps (the store gate, the step limit, the 256-instruction run, faults deep in straight code): every fault program,
# tier against JIT, with and without -fexceptions for the generated code.
# And the failure modes (no compiler, a compiler that fails, a cache that cannot be written): output identical, the JIT never ran.
#
# VGPU_JIT_PROGRAMS="sass_tier.sm_86 sass_tier_steps.sm_86" narrows the programs; VGPU_TIER_WORK keeps the built programs between runs.
#
# The comparison cannot pass by the JIT doing nothing: sass_tier's statistics must show the generated regions ran at least
# VGPU_JIT_MIN_PERCENT (default 10) percent of its warp instructions.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
if [[ -n $(shim_sanitizer "$shim") ]]; then echo "SKIP: the JIT is off in sanitizer builds"; exit 0; fi
command -v c++ >/dev/null 2>&1 || command -v g++ >/dev/null 2>&1 || command -v clang++ >/dev/null 2>&1 || { echo "SKIP: no C++ compiler on PATH"; exit 0; }
if [[ -n ${VGPU_TIER_WORK:-} ]]; then
  work="$VGPU_TIER_WORK"; mkdir -p "$work"
else
  work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-jit.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
fi
export VGPU_TIER_WORK="$work"
build_log="$(VGPU_TIER_BUILD_ONLY=1 "$root/nvidia/tests/e2e/run_sass_tier.sh" 2>&1)"; rc=$?
if grep -q '^SKIP' <<< "$build_log" && ! compgen -G "$work/*.sm_*" >/dev/null; then echo "$build_log" | grep '^SKIP' | head -1; exit 0; fi
(( rc == 0 )) || { echo "FAIL: building the test programs"; echo "$build_log" | tail -20; exit 1; }

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4 [sm_90]=nvidia/h100
                [sm_100]=nvidia/b200 [sm_120]=nvidia/rtx5090)
min_percent="${VGPU_JIT_MIN_PERCENT:-10}"
run_limit="${VGPU_TIER_TIMEOUT:-900}"
cache="$work/jit-cache"
rm -rf "$cache"
fails=0
fault_throws=""

# Runs $bin with the shim, one host thread, counters on, quiet; $1 is the label of the extra environment that follows.
runit() {
  local bin=$1 arch=$2; shift 2
  (cd "$work" && env VGPU_QUIET=1 VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") "$@" timeout "$run_limit" "$bin" 2>&1)
}
jitenv=(VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 "VGPU_JIT_CACHE=$cache")

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
  if [[ $ra != 0 && $p != sass_tier && $ra == "$rb" && "$a" == "$b" ]]; then
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
    elif [[ $p == sass_tier ]]; then
      why="${why:+$why; }the tier's four threads print differently from its one (sass_tier must not)"
    fi
  fi
  if [[ $p == sass_tier ]]; then
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
  fi
  if [[ $p == sass_tier_steps && -z $why ]]; then
    # Faults deep into a run of straight instructions, and the step limit, with the generated code compiled with and without -fexceptions.
    local f fa fb fc thr=0 fsf fenv=(VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_MAX_STEPS=50000 VGPU_QUIET=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim"))
    for f in gather scatter local shared atomic trap forever; do
      fa="$(cd "$work" && env "${fenv[@]}" VGPU_TIER=1 timeout 300 "$bin" fault $f 2>&1)"
      fsf="$work/fault.$f.stats"; rm -f "$fsf"
      fb="$(cd "$work" && env "${fenv[@]}" "${jitenv[@]}" VGPU_JIT_STATS=1 VGPU_JIT_STATS_FILE="$fsf" timeout 300 "$bin" fault $f 2>&1)"
      thr=$(( thr + $(sed -n 's/.* \([0-9][0-9]*\) exceptions unwound.*/\1/p' "$fsf" 2>/dev/null | tail -1 | grep . || echo 0) ))
      fc="$(cd "$work" && env "${fenv[@]}" "${jitenv[@]}" VGPU_JIT_FLAGS=-fno-exceptions "VGPU_JIT_CACHE=$cache-noexc" timeout 300 "$bin" fault $f 2>&1)"
      if ! grep -q "^fault $f: " <<< "$fa" || grep -q "^fault $f: no error" <<< "$fa"; then why="${why:+$why; }the $f fault did not happen"
      elif [[ "$fa" != "$fb" ]]; then why="${why:+$why; }the $f fault is reported differently with the JIT"
      elif [[ "$fa" != "$fc" ]]; then why="${why:+$why; }the $f fault is reported differently with the JIT built without exceptions"; fi
    done
    # The comparison above means something only if faults really were thrown from inside generated regions.
    (( thr >= ${VGPU_JIT_MIN_THROWS:-3} )) || why="${why:+$why; }only $thr of the faults were thrown inside a generated region"
    fault_throws=$thr
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
  [[ -z ${VGPU_JIT_PROGRAMS:-} || " $VGPU_JIT_PROGRAMS " == *" $b "* ]] || continue
  programs+=("${b%.*} ${b##*.}")
done
shopt -u nullglob
(( ${#programs[@]} )) || { echo "SKIP: no test programs were built"; exit 0; }
for j in "${programs[@]}"; do check $j; done

# The failure modes: the JIT cannot do its job and the program must not notice (stdout and stderr, the notices aside: VGPU_QUIET).
prog=sass_tier; arch=sm_86
if [[ -x "$work/$prog.$arch" ]]; then
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
fi

[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: the tier and the JIT differ"
exit $fails
