#!/usr/bin/env bash
# Round sp76, the card part: run on the rig (one A10G), from the root of the source tree, with the corpora in ~/corp
# (tools/bench/real/pack_corpora.sh's layout: the sm_86 binaries built with nvcc 13.0 and -cudart shared, the CUDA Samples under ~/corp/samples).
# Nothing of this was run when the round was written: the round had no card. What each stage should show is in README.md, "The card run".
#   docs/perf-design/data/sp76/card-run.sh <stage> [out dir]        stages: build fork decisions changed keep pf rest, and all (in that order)
# Forms: turbo and exact are this round's rule; turbod0 and exactd0 the same build with VGPU_HOST_GPU_SAMPLE_STARTED=0, the rule before it (no
# launch is sampled once the driver's start has begun). The harness interleaves them, run by run.
# SP76_BUILD: the build directory (default ~/sp76-build; the `build` stage makes it, RelWithDebInfo as the card CI does: for timings name a Release
# build as the summary run's, docs/perf-design/data/summary-2026-11-13/README.md). SP76_CORP: the corpora (default ~/corp).
set -u
stage=${1:?stage: build fork decisions changed keep pf rest all}
out=${2:-$HOME/sp76-out}
build=${SP76_BUILD:-$HOME/sp76-build}
corp=${SP76_CORP:-$HOME/corp}
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
S=$corp/samples/Samples
slow=aligned-types-cuda,babelstream-cuda,base64e-cuda,binomial-cuda
slow_timeout=${SP76_SLOW_TIMEOUT:-90}
# One logged turbo run of a program, cut when it has run `secs` seconds: the mode's lines (the decisions are made before a launch runs).
logged() {   # <tag> <seconds> <cwd> <program and arguments...>
  local tag=$1 secs=$2 cwd=$3
  shift 3
  ( cd "$cwd" && env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 timeout -s KILL "$secs" "$@" > /dev/null 2> "$out/log-$tag.err" )
  grep -E '^host_gpu(_[a-z]+)?: ' "$out/log-$tag.err" | grep -v '^host_gpu:   ' > "$out/log-$tag.txt"
  echo "== $tag"; grep -E 'sampled|starts the driver|host_gpu_nosample|pulled before a sample|host_gpu_kept|launches ran' "$out/log-$tag.txt" | cut -c1-400
}
# Where a CUDA Sample's binary is (the corpus keeps the Samples' tree).
sample_dir() { dirname "$(find "$S" -type f -name "$1" -perm -u+x | head -1)"; }

case $stage in
build|all)
  # The build, the unit tests, e2e_host_gpu (its step 13 is this round's: the process is forked while it holds the real driver, which nothing
  # before this run has done) and e2e_host_gpu_stub (steps 41 to 44).
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  nice cmake --build "$build" -j4 --target vgpucuda > "$build.vgpucuda.log" 2>&1 || tail -20 "$build.vgpucuda.log"
  (cd "$build" && ctest -R 'hostgpu_bench_selftest' --output-on-failure) 2>&1 | tail -3 | tee "$out/selftest.txt"
  ;;&
fork|all)
  # The fork of a process that holds the real driver, twenty times in each mode: `two` of the stand-in test's program (two kernels, the second
  # first launched with the driver up; beside 64 MiB). Every run must print the mode-off output, sample twice with after_driver=1, send both
  # launches, leave no copy; the kernel log must have no Xid and no NVRM line of the runs; and the cost of the fork with the driver loaded
  # (fork_s of the second sample) is read against the fork before the start (fork_s of the first).
  nvcc -std=c++17 -arch=sm_86 -cudart shared -w -Wno-deprecated-gpu-targets nvidia/tests/e2e/host_gpu_lazy.cu nvidia/tests/e2e/host_gpu_lazy_heap.cu -o "$out/lazy" || { echo "FAIL: host_gpu_lazy.cu does not compile"; exit 1; }
  env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" "$out/lazy" two > "$out/fork-off.out" 2> /dev/null
  (dmesg 2> /dev/null || sudo -n dmesg 2> /dev/null) | grep -c -E 'NVRM|Xid' > "$out/fork-dmesg-before.txt"
  : > "$out/fork.txt"
  for mode in exact turbo; do
    ok=0; lost=0
    for i in $(seq 1 20); do
      env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=$mode VGPU_HOST_GPU_SAMPLE=always VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 \
          VGPU_HOST_GPU_START_COST=0 "$out/lazy" two > "$out/fork-$mode.out" 2> "$out/fork-$mode.err"; rc=$?
      line=$(grep '^host_gpu_sampled' "$out/fork-$mode.err")
      forks=$(grep '^host_gpu_sample: ' "$out/fork-$mode.err" | sed -n 's/.* fork_s=\([0-9.]*\) .*/\1/p' | tr '\n' ' ')
      echo "$mode run $i rc=$rc output=$(cmp -s "$out/fork-off.out" "$out/fork-$mode.out" && echo same || echo DIFFERENT) $line fork_s: $forks" >> "$out/fork.txt"
      if [[ $rc -eq 0 ]] && cmp -s "$out/fork-off.out" "$out/fork-$mode.out" && echo "$line" | grep -q 'sampled=2 sent=2 timeouts=0 .* after_driver=1 .* copies_left=0'; then ok=$((ok + 1)); fi
      echo "$line" | grep -q 'timeouts=0' || lost=$((lost + 1))
    done
    echo "$mode: $ok of 20 runs as expected, $lost with a copy that did not answer" | tee -a "$out/fork.txt"
  done
  (dmesg 2> /dev/null || sudo -n dmesg 2> /dev/null) | grep -c -E 'NVRM|Xid' > "$out/fork-dmesg-after.txt"
  echo "kernel log lines with NVRM or Xid: $(cat "$out/fork-dmesg-before.txt") before, $(cat "$out/fork-dmesg-after.txt") after" | tee -a "$out/fork.txt"
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | tee -a "$out/fork.txt"
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the test left: $(ps -eo args= | grep -c "^$out/lazy")"; } | tee -a "$out/fork.txt"
  ;;&
decisions|all)
  # What the mode says of each launch of the round's programs on the card.
  logged aidw 120 "$corp/hec/src/aidw-cuda" ./main 100 1 1 | tee "$out/decisions.txt"
  for p in segmentationTreeThrust histogram mergeSort; do
    logged "$p" 120 "$(sample_dir "$p")" "./$p" | tee -a "$out/decisions.txt"
  done
  logged silu 60 "$corp/hec/src/silu-cuda" ./main 8 8 64 64 1 | tee -a "$out/decisions.txt"
  grep -h -c 'pulled before a sample' "$out"/log-*.txt | tr '\n' ' ' | sed 's/^/pulls before a sample, per program: /' | tee -a "$out/decisions.txt"; echo
  ;;&
changed|all)
  # aidw: 193 s with 3 of 4 launches on the card (Round sp75's card run) to about 45 s with 4 of 4 (its simulator run passes 600 s and is not
  # made: SP76_AIDW_SIM=1 makes it, about 10 minutes). One run a form.
  if [[ -n "${SP76_AIDW_SIM:-}" ]]; then
    $H --corpora hecbench --programs aidw-cuda --runs 1 --long-secs 5 --long-runs 1 --forms sim,turbo,turbod0 --timeout 600 \
       --json "$out/aidw.json" --md "$out/aidw.md" 2>&1 | tee "$out/aidw.txt" | tail -8
  else
    for f in turbo turbod0; do
      e=""; [[ $f == turbod0 ]] && e="VGPU_HOST_GPU_SAMPLE_STARTED=0"
      ( cd "$corp/hec/src/aidw-cuda" && /usr/bin/time -f "aidw $f %e s" env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo $e timeout -s KILL 600 ./main 100 1 1 > "$out/aidw-$f.out" 2> "$out/aidw-$f.err" )
      { tail -1 "$out/aidw-$f.err"; grep -E '^host_gpu: [0-9]+ launches ran|^host_gpu_sampled' "$out/aidw-$f.err"; } | tee -a "$out/aidw.txt"
    done
    cmp -s "$out/aidw-turbo.out" "$out/aidw-turbod0.out" && echo "aidw: the two forms print the same" | tee -a "$out/aidw.txt"
  fi
  # The programs of #659 whose kernels are first launched after the start: ten runs a form.
  $H --corpora samples --programs segmentationTreeThrust,histogram,mergeSort --forms sim,turbo,turbod0 --runs 10 --long-secs 20 --long-runs 3 \
     --json "$out/partly.json" --md "$out/partly.md" 2>&1 | tee "$out/partly.txt" | tail -8
  ;;&
keep|all)
  # The keep list of the integration of sp64-sp68 and the other programs with a sampled launch (data/sp75/census.md). No program more than
  # 3 % slower than its d0 form; outputs as they were.
  $H --corpora polybench,rodinia,hecbench --programs 2MM,3MM,CORR,COVAR,FDTD-2D,LU,SYR2K,SYRK,JACOBI2D,gaussian,lud,hotspot,particlefilter_float,atomicAggregate-cuda,atomicPerf-cuda,black-scholes-cuda,shmembench-cuda,silu-cuda \
     --forms sim,exact,exactd0,turbo,turbod0 --runs 3 --long-secs 20 --json "$out/keep.json" --md "$out/keep.md" 2>&1 | tee "$out/keep.txt" | tail -26
  ;;&
pf|all)
  # Rodinia particlefilter_float (unsettled in Round sp75's card run: turbo 5 to 8 % slower than turbob0). Its normalize_weights_kernel has a
  # race between blocks, so with several host threads it does not compute the same thing twice and its run times fall in two groups, clock
  # seed fixed or not. (1) One host thread, the clock fixed (fixtime.c), six seeds: the two forms take the same time and decide the same.
  # (2) The host's four threads, the clock fixed at one value, 30 runs of turbo, turbo1 (the rule on, the environment of turbob0's size) and
  # turbob0: the two groups, and how each form's runs fall in them. (3) The same at one host thread: one group. (4) As (2) with the clock free.
  cc -shared -fPIC -O1 -o "$out/libfixtime.so" docs/perf-design/data/sp76/fixtime.c
  P="--build $build --corp $corp --stub= --lock $out/pf.lock"
  python3 docs/perf-design/data/sp76/pf.py $P --fixtime "$out/libfixtime.so" --seeds 1763200000:1763200006 --runs 5 --threads 1 2>&1 | tee "$out/pf-fixed.md"
  python3 docs/perf-design/data/sp76/pf_race.py $P --fixtime "$out/libfixtime.so" --threads 4 --runs 30 2>&1 | tee "$out/pf-race4.md" | tail -8
  python3 docs/perf-design/data/sp76/pf_race.py $P --fixtime "$out/libfixtime.so" --threads 1 --runs 5 2>&1 | tee "$out/pf-race1.md" | tail -8
  python3 docs/perf-design/data/sp76/pf_race.py $P --threads 4 --runs 30 2>&1 | tee "$out/pf-race4-free.md" | tail -8
  ;;&
rest|all)
  # The rest of the corpus, one run a form, turbo against turbod0.
  $H --corpora polybench,rodinia,hecbench,samples --skip aidw-cuda,atomicCost-cuda,fpdc-cuda,zerocopy-cuda,GRAMSCHM,tf32TensorCoreGemm,bf16TensorCoreGemm,$slow \
     --runs 1 --forms sim,turbo,turbod0 --timeout 300 --json "$out/rest.json" --md "$out/rest.md" 2>&1 | tee "$out/rest.txt" | tail -30
  $H --corpora hecbench --programs "$slow" --runs 1 --forms sim,turbo,turbod0 --timeout "$slow_timeout" \
     --json "$out/rest-slow.json" --md "$out/rest-slow.md" 2>&1 | tee "$out/rest-slow.txt" | tail -8
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the corpus left: $(ps -eo args= | grep -c "^$corp/")"; } | tee "$out/leftover.txt"
  ;;
esac
