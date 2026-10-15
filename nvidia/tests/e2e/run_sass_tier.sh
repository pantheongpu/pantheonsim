#!/usr/bin/env bash
# The SASS executor's pre-decoded tier (src/sass/tier.hpp, VGPU_TIER=1) against the interpreter (VGPU_TIER=0).
#
# sass_tier.cu (random inputs that break arithmetic through every form the tier has a handler for) is built for each
# architecture below, and so is a set of the other SASS test programs (the ones sass_archs.sh runs on SASS, which
# exercise barriers, divergence, warp ops, atomics, textures, MMA and more). Each runs twice on its generation's
# simulated GPU, with one host thread and the performance counters on: with the interpreter, and with the tier. The two
# must pass and print byte for byte the same -- every result and every counter of every launch, and the text of any
# error. (The tier's statistics, VGPU_TIER_STATS=1, are written to a file: VGPU_TIER_STATS_FILE.)
#
# sass_tier_steps.cu holds what Runner::step_warp does besides executing an instruction -- the store gate, the step count and its
# limit, the 256-instruction run, the count of calls that a warp's turn is made of (a warp that spins on another's store counts
# its iterations), faults -- and the tier's chain of runs is held to all of it. Its faults and its step limit run in processes of
# their own (the text of each error must be the same).
#
# The comparison cannot pass by the tier doing nothing: the statistics must show that the tier ran at least
# VGPU_TIER_MIN_PERCENT (default 30) percent of sass_tier's warp instructions.
#
# VGPU_TIER_OTHER_ARCHS=1 builds the other programs for every architecture too (a soak).
# VGPU_TIER_ARCHS narrows the architectures (default sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120; an nvcc that cannot target one skips it), VGPU_TIER_PROGRAMS the other programs
# (name:first architecture), VGPU_TIER_JOBS caps the builds run at once.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
# VGPU_TIER_WORK=<dir> keeps the built programs there between runs (a program is rebuilt only when its source is newer), for
# working on the tier without compiling fifty programs each time; VGPU_TIER_BUILD_ONLY=1 stops after the builds.
if [[ -n ${VGPU_TIER_WORK:-} ]]; then
  work="$VGPU_TIER_WORK"; mkdir -p "$work"
else
  work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-tier.XXXXXX")"
  trap 'rm -rf "$work"' EXIT
fi

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4 [sm_90]=nvidia/h100
                [sm_100]=nvidia/b200 [sm_120]=nvidia/rtx5090)
archs=(${VGPU_TIER_ARCHS:-sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120})
# program:first architecture[:last]; each runs on one architecture except sass_tier, which runs on every one.
others=(${VGPU_TIER_PROGRAMS:-sass_tier_steps:75 sass_fastpath:75 sass_archs:75 vector_add:75 device_functions:75 device_intrinsics:75 video_forms:75
        symbols:75 surface_oob:75 textures:75 texture_filtering:75 texture_gather:75 texture_layers:75 texture_forms:75
        surface_formatted:75 texture_mipmaps:75 texture_mip_layers:75 texture_srgb:75 texture_int_coords:75 border_colour:75
        block_semaphore:75 cooperative_grid:75 alloca_stack:75 shared_max:75 named_barriers:75 smid:75 smem_size_regs:75
        bsync_recursive_abi:75 waterfall:90 sync_masks:86:86 large_params:75 shared_atomics64:75 mma_forms:80 mma_fragment_layout:80
        modern_dtypes:80 wmma_gemm:80 wmma_types:80 vector_atomics:90 dsmem_cluster:90 cooperative_cluster:90})
min_percent="${VGPU_TIER_MIN_PERCENT:-30}"

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
read -r rmaj rmin <<< "$("$nvcc_bin" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
# sync_masks' expected values are CUDA 13's ptxas code; the others need nothing special.
jobs=()
for arch in "${archs[@]}"; do
  if ! grep -qx "$arch" <<< "$supported"; then echo "SKIP $arch: this nvcc cannot target it"; continue; fi
  jobs+=("sass_tier $arch")
done
for p in "${others[@]}"; do
  IFS=: read -r name first last <<< "$p"
  [[ $name == sync_masks && ${rmaj:-0} -lt 13 ]] && continue
  [[ $name == mma_forms && ${rmaj:-0} -lt 13 ]] && continue   # its sparse forms are more than CUDA 12.0's ptxas assembles
  [[ $name == vector_atomics ]] && (( ${rmaj:-0} * 100 + ${rmin:-0} < 1201 )) && continue
  # One architecture each (sm_86, or sm_90 for the Hopper programs), unless VGPU_TIER_OTHER_ARCHS=1 (a soak: every one).
  if [[ -n ${VGPU_TIER_OTHER_ARCHS:-} ]]; then oarchs=("${archs[@]}"); elif (( first > 86 )); then oarchs=(sm_90); else oarchs=(sm_86); fi
  for arch in "${oarchs[@]}"; do
    (( ${arch#sm_} >= first && ${arch#sm_} <= ${last:-999} )) || continue
    grep -qx "$arch" <<< "$supported" && jobs+=("$name $arch")
  done
done
(( ${#jobs[@]} )) || { echo "SKIP: no architecture this nvcc can target"; exit 0; }

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
export nvcc_bin root work
export san="${san_flags[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "${VGPU_TIER_JOBS:-$("$root/ci/jobs.sh" -p sass)}" -L 1 bash -c '
  [[ -x "$work/$0.$1" && "$work/$0.$1" -nt "$root/nvidia/tests/e2e/$0.cu" && ! -s "$work/$0.$1.build" ]] && exit 0
  rm -f "$work/$0.$1" "$work/$0.$1.build"
  "$nvcc_bin" -std=c++17 -arch="$1" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
    -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.$1" -lcuda 2> "$work/$0.$1.build" ||
    echo "build failed" >> "$work/$0.$1.build"'

[[ -n ${VGPU_TIER_BUILD_ONLY:-} ]] && { echo "built in $work"; exit 0; }

fails=0
check() {
  local p=$1 arch=$2 bin="$work/$1.$2" a b ra rb why=""
  if [[ ! -x $bin ]]; then
    echo "FAIL $arch $p: does not build"; sed 's/^/    /' "$bin.build" | tail -5; fails=1; return
  fi
  require_shim_libs "$shim" "$bin" >/dev/null || return
  local env=(VGPU_QUIET=1 VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim"))
  a="$(cd "$work" && env "${env[@]}" VGPU_TIER=0 timeout 900 "$bin" 2>&1)"; ra=$?
  # (The tier's statistics go to a file of their own: printed at exit among the program's buffered output, they tear a line.)
  local stats bq sf="$work/$p.$arch.stats"
  rm -f "$sf"
  b="$(cd "$work" && env "${env[@]}" VGPU_TIER=1 VGPU_TIER_STATS=1 VGPU_TIER_STATS_FILE="$sf" timeout 900 "$bin" 2>&1)"; rb=$?
  stats="$(grep -h '^tier: ' "$sf" 2>/dev/null | head -1)"
  bq="$b"
  grep -q "running SASS" <<< "$a" || why="its SASS did not run"
  grep -q "running PTX instead of SASS" <<< "$a" && why="fell back to PTX: $(grep -m1 'running PTX instead' <<< "$a")"
  [[ $ra == "$rb" ]] || why="${why:+$why; }exit status $ra with the interpreter, $rb with the tier"
  [[ "$a" == "$bq" ]] || why="${why:+$why; }the interpreter and the tier print differently"
  if [[ $ra != 0 && $p != sass_tier && $ra == "$rb" && "$a" == "$bq" ]]; then
    # A program that fails the same way with the interpreter has a failure that is not the tier's (another test owns it).
    echo "ok   $arch $p  (fails identically in both, exit $ra: not compared further)"
    return
  fi
  [[ $ra == 0 ]] || why="${why:+$why; }the run failed (exit $ra)"
  if [[ $p == sass_tier ]]; then
    # The store gate (step_warp) is what this line checks, in both runs: four warps loading and storing one word end with 1.
    grep -q '^store gate: .* ok$' <<< "$a" || why="${why:+$why; }the interpreter's store gate line is not ok"
    grep -q '^store gate: .* ok$' <<< "$bq" || why="${why:+$why; }the tier's store gate line is not ok"
    # The tier is shared by the host threads of a launch: four of them must give what one thread of the interpreter does.
    local c4
    c4="$(cd "$work" && env "${env[@]}" VGPU_TIER=1 VGPU_THREADS=4 timeout 900 "$bin" 2>&1)"
    [[ "$a" == "$c4" ]] || why="${why:+$why; }four threads with the tier print differently from the interpreter's one"
    local ran total
    ran="$(sed -n 's/^tier: \([0-9]*\) of \([0-9]*\) warp.*/\1/p' <<< "$stats")"
    total="$(sed -n 's/^tier: \([0-9]*\) of \([0-9]*\) warp.*/\2/p' <<< "$stats")"
    if [[ -z $ran || -z $total ]] || (( total == 0 || ran * 100 < total * min_percent )); then
      why="${why:+$why; }the tier ran ${ran:-?} of ${total:-?} warp instructions (wanted at least $min_percent%)"
    fi
  fi
  if [[ ( $p == sass_tier_steps || $p == sass_tier ) && -z $why ]]; then
    # The tier without its chain (one run of straight instructions per call of the scheduler, as the interpreter has it): the
    # same again, so that a difference of the chain's is told from one of the handlers'.
    local nc
    nc="$(cd "$work" && env "${env[@]}" VGPU_TIER=1 VGPU_TIER_CHAIN=0 timeout 900 "$bin" 2>&1)"
    [[ "$a" == "$nc" ]] || why="${why:+$why; }the tier without its chain prints differently from the interpreter"
  fi
  if [[ $p == sass_tier_steps && -z $why ]]; then
    # Faults deep into a run of straight instructions, and the step limit: the text names the instruction, the lane and the
    # address, so a handler that reports it differently shows here. VGPU_MAX_STEPS keeps the loop with no end short.
    local f fa fb fenv=(VGPU_GPU=${gpu[$arch]} VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_MAX_STEPS=50000 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim"))
    for f in gather scatter local shared atomic trap forever; do
      fa="$(cd "$work" && env "${fenv[@]}" VGPU_TIER=0 timeout 300 "$bin" fault $f 2>&1)"
      fb="$(cd "$work" && env "${fenv[@]}" VGPU_TIER=1 timeout 300 "$bin" fault $f 2>&1)"
      if ! grep -q "^fault $f: " <<< "$fa" || grep -q "^fault $f: no error" <<< "$fa"; then why="${why:+$why; }the $f fault did not happen"
      elif [[ "$fa" != "$fb" ]]; then why="${why:+$why; }the $f fault is reported differently by the tier"; fi
    done
  fi
  if [[ -n $why ]]; then
    echo "FAIL $arch $p: $why"
    diff <(echo "$a") <(echo "$bq") | head -12 | sed 's/^/    /'
    fails=1
  else
    echo "ok   $arch $p${stats:+  ($stats)}"
  fi
}
for j in "${jobs[@]}"; do check $j; done
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: the tier and the interpreter differ"
exit $fails
