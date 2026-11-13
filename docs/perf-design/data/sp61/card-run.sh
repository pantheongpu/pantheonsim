#!/usr/bin/env bash
# Round sp61, the card part: run on the rig (one A10G), from the root of the source tree, after the corpora are in ~/corp
# (tools/bench/real/pack_corpora.sh's layout, the sm_86 binaries built with nvcc 13.0 and -cudart shared). Built on sp60b's card-run.sh.
#   docs/perf-design/data/sp61/card-run.sh <stage> [out dir]        stages: build ci target ab log all, and repeat (not part of all)
# Forms: exact and turbo sample inside small grids (the default of this round); exactw and turbow sample whole blocks only
# (VGPU_HOST_GPU_SAMPLE=whole: the rule of Round sp60b, the bar's reference). Interleaved, run by run, with the plain simulator.
set -u
stage=${1:?stage}
out=${2:-$HOME/sp61-out}
root=$(pwd)
build=$HOME/sp61-build
corp=$HOME/corp
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
HEC_SKIP="aidw-cuda,atomicCost-cuda,fpdc-cuda,zerocopy-cuda"
# the programs with a launch that is a candidate inside its blocks in the offline pass (data/sp61/offline-turbo.md): the only ones whose
# decisions can differ from sp60b's rule. Only COVAR and CORR (the target) passed the floor there; the others are short and are here so that the
# A/B sees it if the rig's ages or memory let one through.
TARGET=polybench/COVAR,polybench/CORR
INSIDE=${SP61_INSIDE:-$TARGET,polybench/ATAX,polybench/BICG,polybench/GEMVER,polybench/GESUMMV,polybench/MVT,rodinia/sc_gpu,hecbench/aop-cuda,hecbench/bh-cuda}

case $stage in
build|all)
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  ;;&
ci|all)
  # the stand-in-driver test (steps 21 to 23 are new and have run nowhere), the logic test and the harness's self-test
  (cd "$build" && ctest -R 'e2e_host_gpu_stub|test_host_gpu_logic|hostgpu_bench_selftest' --output-on-failure) 2>&1 | tee "$out/stub.txt"
  ;;&
target|all)
  # COVAR and CORR: 3 runs of each form (about 60 s a run in the simulator on the rig, a few seconds on the card)
  $H --programs "$TARGET" --runs 3 --long-secs 1000 --forms sim,exact,exactw,turbo,turbow --timeout 600 \
     --json "$out/target.json" --md "$out/target.md" 2>&1 | tee "$out/target.txt" | tail -20
  ;;&
ab|all)
  # the bar: every program with a launch sampled inside its blocks, 3 runs a form against sp60b's rule; then the rest of the corpus one run a form
  # (their decisions are sp60b's by construction: no launch of theirs is a candidate inside its blocks)
  $H --programs "$INSIDE" --runs 3 --long-secs 30 --long-runs 1 --forms sim,exact,exactw,turbo,turbow --timeout 600 \
     --json "$out/ab-inside.json" --md "$out/ab-inside.md" 2>&1 | tee "$out/ab-inside.txt" | tail -30
  $H --corpora polybench,rodinia --skip "GRAMSCHM,2MM,LU,FDTD-2D,COVAR,CORR,ATAX,BICG,GEMVER,GESUMMV,MVT,sc_gpu" --runs 1 --forms sim,exact,exactw,turbo,turbow --timeout 300 \
     --json "$out/ab-rest.json" --md "$out/ab-rest.md" 2>&1 | tee "$out/ab-rest.txt" | tail -30
  ;;&
log|all)
  # what each sample inside the blocks measured on the rig's CPU, and what the card then took
  for p in COVAR CORR; do
    for m in exact turbo; do
      echo "== $p $m"
      ( cd "$corp/polybench/bin" && env VGPU_QUIET=1 VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$build/shim VGPU_HOST_GPU=$m VGPU_HOST_GPU_LOG=1 timeout 600 ./$p 2>&1 \
        | grep -E "^host_gpu(_sample|_sampled|_turbo|: .*(sent by|on the card|failed))" | cut -c1-900 )
    done
  done > "$out/sample-log.txt" 2>&1
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the corpus left: $(ps -eo args= | grep -c "^$corp/")"; } | tee "$out/leftover.txt"
  ;;
repeat)
  # ten more runs of each form of any program the A/B shows more than 3% apart
  $H --programs "${SP61_REPEAT:?programs}" --runs 10 --forms sim,exact,exactw,turbo,turbow --timeout 300 --no-exact-stats \
     --json "$out/repeat.json" --md "$out/repeat.md" 2>&1 | tee "$out/repeat.txt" | tail -12
  ;;
esac
