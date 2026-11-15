#!/usr/bin/env bash
# Round sp76: mutants of the rule for a sample once the driver is up, where the unit test cannot see it (the mode's decision to sample, its
# plan at the fork, the pull before the fork, the switch). Each is the tree with one change, built in a build directory of its own;
# nvidia/tests/e2e/run_host_gpu_stub.sh (steps 41 to 44) must fail against it. Builds and runs hold the machine's build lock.
# (The runner is round sp75's e2e-mutants.sh with this list.)
#   e2e-mutants.sh <tree> <work dir outside the tree> <nvcc> [only]
set -u
src=$(cd "$1" && pwd); work=$2; nvcc=$3; only=${4:-}
mkdir -p "$work"
lock=/tmp/pantheonsim-builder.lock
names=("mode: no launch is a candidate for a sample once the driver has been started (the rule before)"
       "mode: the fork is refused once the driver has been started"
       "mode: VGPU_HOST_GPU_SAMPLE_STARTED=0 not read"
       "mode: nothing is pulled before the fork"
       "mode: the pull before the fork leaves the chunks the card's alone")
files=(src/runtime/host_gpu.cpp src/runtime/host_gpu.cpp src/runtime/host_gpu.cpp src/runtime/host_gpu.cpp src/runtime/host_gpu.cpp)
olds=('  if (!sample_allowed(start_sh != nullptr, start_ready, start_failed, sample_started)) return early("the driver has been started");'
      '    if (!sample_allowed(impl_->start_sh != nullptr, impl_->start_ready, impl_->start_failed, impl_->sample_started)) {'
      'v && std::string(v) == "0") r->impl_->sample_started = false;'
      '      const uint64_t pulled = impl_->pull_for_copy();'
      '      pull_chunks(a, a.r.sim_read(0, a.r.chunks() - 1));')
news=('  if (start_sh || start_failed) return early("the driver has been started");'
      '    if (impl_->start_sh || impl_->start_failed) {'
      'v && std::string(v) == "0") r->impl_->sample_started = true;'
      '      const uint64_t pulled = 0;'
      '      pull_chunks(a, {});')
# (Mutants 3 and 4 are killed through the copy's own check, sample_copy_begin: a copy that finds a chunk only the card holds ends without an
# answer. That check and the refusal of the gate's pull in a copy have no mutant of their own: in a process with one launching thread they
# never act, and the stand-in card computes nothing, so a copy that read stale memory would answer the same.)
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
    flock "$lock" timeout 150 nice -n 19 ninja -C "$work/build" -j3 vgpucudart vgpucuda > "$work/build.log" 2>&1; rc=$?
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
