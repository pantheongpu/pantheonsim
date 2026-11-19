#!/usr/bin/env bash
# Differential execution and determinism of the NVIDIA engines in exact mode.
#
#   run_differential.sh diff          SASS against PTX
#   run_differential.sh determinism   one program, many runs
#
# Both modes build each program below for each architecture (its SASS, with its PTX beside it) and run it on that
# generation's simulated GPU with the performance counters, the per-launch memory digest and the program's own output
# recorded.
#
# diff: one run on the SASS engine and one on the PTX interpreter (VGPU_SASS=0), one host thread each. Both must exit 0,
# print the same bytes, leave the same bytes in every allocation after every launch (VGPU_KERNEL_DIGEST, with where the
# allocations lie dropped), and launch the same kernels with the same grid and block, in the same order, with the same
# counters. A counter is only compared when both engines define it the same way: the instruction counts are of
# different instruction sets (PTX's, SASS's), and the load counts, requests, sectors and bank conflicts follow what
# ptxas made of the loads (it merges, widens and drops them), so those are listed, not failed. What each engine writes
# is the program's own: bytes written to global and shared memory (nothing a compiler may merge changes how many bytes
# a program stores), blocks and warps are compared. VGPU_DIFF_INFO=1 prints how often each other counter differs.
#
# determinism: the same program run VGPU_DET_REPS times (default 5; 20 in the nightly job) on one host thread, with the
# process pinned to the first CPU, to the last CPU, to every CPU, niced, with a different environment size and working
# directory, and once with VGPU_EXACT_THREADS=4 (the promise of docs/exact-threads.md: the same as one thread). Output,
# every counter of every launch and the memory digests must be the same in all of them. On the PTX engine the random
# and the adversarial warp scheduler with one seed are run twice each and must replay.
#
# VGPU_NIGHTLY=1 runs the full program and architecture lists; without it a short list that fits a pull request.
# VGPU_DIFF_ARCHS and VGPU_DIFF_PROGRAMS (name:first architecture[:last]) narrow either; VGPU_DIFF_JOBS caps the builds.
set -uo pipefail
mode="${1:-diff}"
case "$mode" in diff|determinism) ;; *) echo "usage: $0 diff|determinism"; exit 2 ;; esac
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
if ! grep -q VGPU_SASS_LOG "$root/nvidia/src/fatbin.cpp" 2>/dev/null; then echo "SKIP: this simulator has no SASS path"; exit 0; fi
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-diffdet.XXXXXX")"
keep=0
trap '[[ $keep == 1 && -n ${VGPU_DIFF_KEEP:-} ]] && cp -r "$work" "$VGPU_DIFF_KEEP"; rm -rf "$work"' EXIT

declare -A gpu=([sm_75]=nvidia/t4 [sm_80]=nvidia/a100 [sm_86]=nvidia/rtx3060 [sm_89]=nvidia/l4
                [sm_90]=nvidia/h100 [sm_100]=nvidia/b200 [sm_120]=nvidia/rtx5090)
if [[ -n ${VGPU_NIGHTLY:-} ]]; then
  def_archs="sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120"
  def_progs="sass_archs:75 vector_add:75 device_functions:75 device_intrinsics:75 video_forms:75 dpx_intrinsics:75 minmax_forms:80
    symbols:75 surface_oob:75 textures:75 texture_filtering:75 texture_gather:75 texture_layers:75 texture_forms:75
    surface_formatted:75 texture_mipmaps:75 texture_mip_layers:75 texture_srgb:75 texture_int_coords:75 border_colour:75
    block_semaphore:75 cooperative_grid:75 alloca_stack:75 managed_vars:75 shared_max:75 named_barriers:75 smid:75
    smem_size_regs:75 bsync_recursive_abi:75 dynamic_parallelism:75 cdp_device_api:75 rdc_device_api:75 large_params:75
    shared_atomics64:75 waterfall:90 mma_forms:80 mma_fragment_layout:80 modern_dtypes:80 wmma_gemm:80 wmma_types:80
    vector_atomics:90 dsmem_cluster:90 cooperative_cluster:90 atomic_cas:75 atomics_threads:75 lockstep_blocks:75
    div_approx:75 float_half_forms:75 ldmatrix_forms:80 lop3_pred_chain:75 mul_add_contraction:75 packed_int_forms:75
    ptx_half_forms:75 ret_condition_predicate:75 group_ballot_barrier:75 bar_red_named:75 constant_global_load:75
    constant_table_xor:75 device_heap:75 jit_loops:75 narrow_cvt:100 packed_f32x2:100:100"
else
  def_archs="sm_86 sm_90"
  def_progs="vector_add:75 device_intrinsics:75 atomic_cas:75 block_semaphore:75 cooperative_grid:75 alloca_stack:75
    named_barriers:75 textures:75 wmma_gemm:80 dsmem_cluster:90"
fi
archs=(${VGPU_DIFF_ARCHS:-$def_archs})
progs=(${VGPU_DIFF_PROGRAMS:-$def_progs})
rdc=" dynamic_parallelism cdp_device_api rdc_device_api "
two_devices=" managed_vars runtime_conformance "

supported="$("$nvcc_bin" --list-gpu-code 2>/dev/null)"
release="$("$nvcc_bin" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
read -r rmaj rmin <<< "$release"
drop_prog() { local kept=() p; for p in "${progs[@]}"; do [[ ${p%%:*} == "$1" ]] || kept+=("$p"); done; progs=("${kept[@]}"); }
if (( ${rmaj:-0} < 13 )); then echo "SKIP mma_forms: its SASS needs CUDA 13's ptxas"; drop_prog mma_forms; fi
if (( ${rmaj:-0} * 100 + ${rmin:-0} < 1201 )); then drop_prog vector_atomics; fi
jobs=()
for arch in "${archs[@]}"; do
  if ! grep -qx "$arch" <<< "$supported"; then echo "SKIP $arch: this nvcc cannot target it"; continue; fi
  for p in "${progs[@]}"; do
    IFS=: read -r name first last <<< "$p"
    [[ -f "$root/nvidia/tests/e2e/$name.cu" ]] || { echo "SKIP $name: no such program"; continue; }
    (( ${arch#sm_} >= first && ${arch#sm_} <= ${last:-999} )) && jobs+=("$name $arch")
  done
done
(( ${#jobs[@]} )) || { echo "SKIP: nothing this nvcc can build"; exit 0; }

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
export nvcc_bin root work rdc
export san="${san_flags[*]}"
printf '%s\n' "${jobs[@]}" | xargs -P "${VGPU_DIFF_JOBS:-$("$root/ci/jobs.sh" -p sass)}" -L 1 bash -c '
  sep=(); [[ $rdc == *" $0 "* ]] && sep=(-rdc=true -lcudadevrt)
  "$nvcc_bin" -std=c++17 -arch="$1" -cudart shared -w -Wno-deprecated-gpu-targets -Xlinker --as-needed $san \
    -I"$root/nvidia/tests/e2e" "$root/nvidia/tests/e2e/$0.cu" -o "$work/$0.$1" -lcuda "${sep[@]}" 2> "$work/$0.$1.build" ||
    echo "build failed" >> "$work/$0.$1.build"'

fails=0
# Where the allocations lie is not part of the comparison (the two engines may lay memory out differently).
strip_addr() { sed -E 's/ [0-9a-f]+:/ /g' "$1"; }
# The counter records of a run, one line a launch, reduced to the named keys (the launch number and the pid differ by design).
# $1 file, rest the keys; "kernel", "grid" and "block" are always kept.
cnt_keys() {
  local f=$1; shift
  awk -v keys="kernel grid block $*" 'BEGIN { n = split(keys, K, " ") }
    { out = ""
      for (i = 1; i <= n; i++)
        if (match($0, "\"" K[i] "\":(\"[^\"]*\"|\\[[^]]*\\]|[0-9]+)")) out = out substr($0, RSTART, RLENGTH) " "
      print out }' "$f"
}
all_keys="blocks warps instructions thread_instructions global_loads global_stores global_bytes_read global_bytes_written
  shared_loads shared_stores shared_bytes_read shared_bytes_written global_sectors_ld global_sectors_st global_requests_ld
  global_requests_st shared_requests_ld shared_requests_st shared_bank_conflicts_ld shared_bank_conflicts_st global_bytes_ld global_bytes_st"
same_keys="blocks warps global_bytes_written shared_bytes_written"

# run_one <program> <arch> <tag> <env...> -- <prefix command words...>
# Writes $bin.<tag>.out (program output), .cnt (counter records), .dig (digest), .rc.
run_one() {
  local p=$1 arch=$2 tag=$3; shift 3
  local bin="$work/$p.$arch" envs=() pre=()
  while (( $# )) && [[ $1 != -- ]]; do envs+=("$1"); shift; done
  shift
  pre=("$@")
  rm -f "$bin.$tag.cnt" "$bin.$tag.dig"
  local base=(VGPU_QUIET=1 VGPU_GPU="${gpu[$arch]}" LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim")
              VGPU_COUNTERS_FILE="$bin.$tag.cnt" VGPU_KERNEL_DIGEST="$bin.$tag.dig")
  [[ $two_devices == *" $p "* ]] && base+=(VGPU_DEVICE_COUNT=2)
  (cd "$work" && env "${base[@]}" "${envs[@]}" "${pre[@]}" timeout 900 "$bin" > "$bin.$tag.out" 2>&1); echo $? > "$bin.$tag.rc"
  touch "$bin.$tag.cnt" "$bin.$tag.dig"
}
runs=0 compared=0 fellback=0
declare -A differ   # counter -> launches that differ, over every program (VGPU_DIFF_INFO)

check_diff() {
  local p=$1 arch=$2 bin="$work/$1.$2" why=""
  run_one "$p" "$arch" sass VGPU_THREADS=1 VGPU_SASS_LOG=1 --
  run_one "$p" "$arch" ptx VGPU_THREADS=1 VGPU_SASS=0 --
  local rs rp; rs=$(<"$bin.sass.rc"); rp=$(<"$bin.ptx.rc")
  if ! grep -q "running SASS" "$bin.sass.out" || grep -q "running PTX instead of SASS" "$bin.sass.out"; then
    echo "note $arch $p: ran PTX both times ($(grep -m1 -o 'running PTX instead of SASS.*' "$bin.sass.out" || echo 'no SASS ran'))"
    fellback=$((fellback + 1)); return
  fi
  [[ $rs == 0 ]] || why="the SASS run exited $rs"
  [[ $rp == 0 ]] || why="${why:+$why; }the PTX run exited $rp"
  cmp -s <(grep -v '^\[vgpu\] running' "$bin.sass.out") "$bin.ptx.out" || why="${why:+$why; }the output differs"
  cmp -s <(strip_addr "$bin.sass.dig") <(strip_addr "$bin.ptx.dig") || why="${why:+$why; }memory after a launch differs"
  # The launches themselves: same kernels, grids and blocks in the same order, and the counters both engines define alike.
  cnt_keys "$bin.sass.cnt" $same_keys > "$bin.sass.cmp"
  cnt_keys "$bin.ptx.cnt" $same_keys > "$bin.ptx.cmp"
  cmp -s "$bin.sass.cmp" "$bin.ptx.cmp" || why="${why:+$why; }counters differ"
  if [[ -n ${VGPU_DIFF_INFO:-} ]]; then
    local k a b
    for k in $all_keys; do
      a=$(cnt_keys "$bin.sass.cnt" "$k" | paste -sd' '); b=$(cnt_keys "$bin.ptx.cnt" "$k" | paste -sd' ')
      [[ $a == "$b" ]] || { differ[$k]=$(( ${differ[$k]:-0} + 1 )); echo "info $arch $p: $k differs between SASS and PTX (not compared)"; }
    done
  fi
  compared=$((compared + 1))
  if [[ -n $why ]]; then
    echo "FAIL $arch $p: $why"; keep=1
    diff <(grep -v '^\[vgpu\] running' "$bin.sass.out") "$bin.ptx.out" | head -6 | cut -c1-200 | sed 's/^/    out: /'
    diff <(strip_addr "$bin.ptx.dig") <(strip_addr "$bin.sass.dig") | head -4 | cut -c1-200 | sed 's/^/    digest: /'
    diff "$bin.ptx.cmp" "$bin.sass.cmp" | head -6 | cut -c1-240 | sed 's/^/    counters (ptx <, sass >): /'
    fails=1
  else
    echo "ok   $arch $p ($(wc -l < "$bin.ptx.cnt") launches)"
  fi
}

# One run of the determinism set: its identity is output, every counter and the digest.
signature() {
  local bin=$1 tag=$2
  { cat "$bin.$tag.rc"
    grep -v '^\[vgpu\] running' "$bin.$tag.out"
    cnt_keys "$bin.$tag.cnt" $all_keys
    strip_addr "$bin.$tag.dig"; } > "$bin.$tag.sig"
}

check_determinism() {
  local p=$1 arch=$2 bin="$work/$1.$2" why="" reps="${VGPU_DET_REPS:-5}" i ncpu last
  ncpu=$(nproc); last=$((ncpu - 1))
  run_one "$p" "$arch" r0 VGPU_THREADS=1 VGPU_SASS_LOG=1 --
  if ! grep -q "running SASS" "$bin.r0.out" || grep -q "running PTX instead of SASS" "$bin.r0.out"; then
    echo "note $arch $p: ran PTX, no SASS to run twice"; fellback=$((fellback + 1))
  fi
  signature "$bin" r0
  [[ $(<"$bin.r0.rc") == 0 ]] || why="the first run exited $(<"$bin.r0.rc")"
  for ((i = 1; i < reps; i++)); do
    local pre=() envs=(VGPU_THREADS=1 VGPU_SASS_LOG=1)
    case $((i % 5)) in
      1) command -v taskset >/dev/null && pre=(taskset -c 0) ;;
      2) command -v taskset >/dev/null && pre=(taskset -c "$last") ;;
      3) pre=(nice -n 10) ;;
      4) envs+=("CI_ENV_PADDING_$i=$(printf 'x%.0s' $(seq 1 $((i * 997))))") ;;
      0) command -v taskset >/dev/null && pre=(taskset -c "0-$last") ;;
    esac
    run_one "$p" "$arch" r$i "${envs[@]}" -- "${pre[@]}"
    signature "$bin" r$i
    cmp -s "$bin.r0.sig" "$bin.r$i.sig" || { why="${why:+$why; }run $i (${pre[*]:-env pad}) differs from run 0"; diff "$bin.r0.sig" "$bin.r$i.sig" | head -6 | cut -c1-240 | sed 's/^/    /'; }
  done
  # The promise of docs/exact-threads.md: four host threads that prove the blocks independent give one thread's answer.
  run_one "$p" "$arch" ex VGPU_THREADS=1 VGPU_EXACT_THREADS=4 VGPU_SASS_LOG=1 --
  signature "$bin" ex
  cmp -s "$bin.r0.sig" "$bin.ex.sig" || { why="${why:+$why; }VGPU_EXACT_THREADS=4 differs from one thread"; diff "$bin.r0.sig" "$bin.ex.sig" | head -6 | cut -c1-240 | sed 's/^/    /'; }
  # PTX, replayable orders: the same seed is the same run.
  local sched
  for sched in random adversarial; do
    run_one "$p" "$arch" "${sched}1" VGPU_THREADS=1 VGPU_SASS=0 VGPU_SCHEDULER=$sched VGPU_SCHEDULER_SEED=7 --
    run_one "$p" "$arch" "${sched}2" VGPU_THREADS=1 VGPU_SASS=0 VGPU_SCHEDULER=$sched VGPU_SCHEDULER_SEED=7 --
    signature "$bin" "${sched}1"; signature "$bin" "${sched}2"
    cmp -s "$bin.${sched}1.sig" "$bin.${sched}2.sig" || { why="${why:+$why; }the $sched scheduler with one seed did not replay"; diff "$bin.${sched}1.sig" "$bin.${sched}2.sig" | head -6 | cut -c1-240 | sed 's/^/    /'; }
  done
  compared=$((compared + 1))
  if [[ -n $why ]]; then echo "FAIL $arch $p: $why"; keep=1; fails=1
  else echo "ok   $arch $p ($reps runs, exact threads, 2 schedulers)"; fi
}

for j in "${jobs[@]}"; do
  read -r p arch <<< "$j"
  bin="$work/$p.$arch"
  if [[ ! -x $bin ]]; then
    echo "FAIL $arch $p: does not build"; tail -5 "$bin.build" | sed 's/^/    /'; fails=1; continue
  fi
  require_shim_libs "$shim" "$bin" >/dev/null || { echo "SKIP $arch $p: needs a library this shim lacks"; continue; }
  runs=$((runs + 1))
  if [[ $mode == diff ]]; then check_diff "$p" "$arch"; else check_determinism "$p" "$arch"; fi
done
if [[ -n ${VGPU_DIFF_INFO:-} ]]; then
  for k in "${!differ[@]}"; do echo "info: $k differs between SASS and PTX in ${differ[$k]} program runs"; done
fi
echo "$mode: $compared of $runs program/architecture runs compared; $fellback ran PTX only"
if (( compared == 0 )); then echo "FAIL: nothing was compared"; fails=1; fi
[[ $fails == 0 ]] && echo PASS || echo "FAIL: $mode"
exit $fails
