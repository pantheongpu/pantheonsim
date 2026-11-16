#!/usr/bin/env bash
# Round sp75: mutants of the bounded-block rule where the unit test cannot see it (the engine's part loop, the forked copy's answer, the mode's plan
# and its switch). Each is the tree with one change, built in a build directory of its own; nvidia/tests/e2e/run_host_gpu_stub.sh (steps 38 to 40)
# must fail against it. Builds and runs hold the machine's build lock.
#   e2e-mutants.sh <tree> <work dir outside the tree> <nvcc> [only]
set -u
src=$(cd "$1" && pwd); work=$2; nvcc=$3; only=${4:-}
mkdir -p "$work"
lock=/tmp/pantheonsim-builder.lock
names=("engine: the limit never stops a block" "engine: a stopped block's timed run goes on to its end" "mode: the plan sets no limit" "mode: VGPU_HOST_GPU_SAMPLE_BLOCK=0 not read"
       "copy: the stopped block not reported" "mode: a small grid planned as a sampled one")
files=(src/sass/exec.cpp src/sass/exec.cpp src/runtime/host_gpu.cpp src/runtime/host_gpu.cpp src/runtime/runtime.cpp src/runtime/host_gpu.cpp)
olds=('    if (since(first) >= part_block_limit_s_) {' '    for (uint64_t made = 0; made < rounds && block_round(blk, kBoundedSlice); ++made) {' '  if (t_sample_hint.bounded) {'
      'v && std::string(v) == "0") r->impl_->sample_block = false;' '      msg.unfinished = part.unfinished ? 1 : 0;'
      '  plan->limit = inside ? inblock_blocks(t_sample_hint.blocks) : small ? bounded_blocks(t_sample_hint.blocks) : sample_blocks(t_sample_hint.blocks);')
news=('    if (false) {' '    for (uint64_t made = 0; block_round(blk, kBoundedSlice); ++made) {' '  if (false) {'
      'v && std::string(v) == "0") r->impl_->sample_block = true;' '      msg.unfinished = 0;'
      '  plan->limit = inside ? inblock_blocks(t_sample_hint.blocks) : sample_blocks(t_sample_hint.blocks);')
killed=0; ran=0
for i in "${!names[@]}"; do
  [[ -z "$only" || "$only" == "$i" ]] || continue
  rm -rf "$work/src"; mkdir -p "$work/src"
  (cd "$src" && tar -cf - --exclude=.git --exclude='docs/perf-design/data' .) | tar -C "$work/src" -xf -
  for f in "${files[@]}"; do touch "$work/src/$f"; done   # (so that the file the mutant before this one changed is compiled again)
  python3 - "$work/src/${files[$i]}" "${olds[$i]}" "${news[$i]}" <<'PY' || { echo "$i NOT APPLIED: ${names[$i]}"; continue; }
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
sys.exit(1) if s.count(old) != 1 else open(p, "w").write(s.replace(old, new))
PY
  ran=$((ran + 1))
  [[ -f "$work/build/build.ninja" ]] || (export PATH=$(dirname "$nvcc"):$PATH CUDA_HOME=$(dirname "$(dirname "$nvcc")"); cmake -S "$work/src" -B "$work/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DVGPU_LTO=OFF > "$work/cmake.log" 2>&1)
  while :; do
    flock "$lock" timeout 270 nice -n 19 ninja -C "$work/build" -j3 vgpucudart vgpucuda > "$work/build.log" 2>&1; rc=$?
    [[ $rc -eq 124 ]] || break
  done
  if [[ $rc -ne 0 ]]; then echo "$i killed (does not build): ${names[$i]}"; killed=$((killed + 1)); continue; fi
  flock "$lock" env VGPU_NVCC="$nvcc" VGPU_BUILD_DIR="$work/build" VGPU_HOST_GPU_TEST_PROFILE=nvidia/a10g nice -n 19 bash "$work/src/nvidia/tests/e2e/run_host_gpu_stub.sh" > "$work/e2e-$i.out" 2>&1; rc=$?
  if [[ $rc -ne 0 ]] && grep -q '^FAIL' "$work/e2e-$i.out"; then
    killed=$((killed + 1))
    echo "$i killed by e2e_host_gpu_stub ($(grep -c '^FAIL' "$work/e2e-$i.out") lines; first: $(grep '^FAIL' "$work/e2e-$i.out" | head -1 | cut -c1-150)): ${names[$i]}"
  else
    echo "$i SURVIVED (rc $rc): ${names[$i]}"
  fi
done
echo "$killed of $ran killed"
