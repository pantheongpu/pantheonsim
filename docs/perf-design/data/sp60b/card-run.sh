#!/usr/bin/env bash
# Round sp60b, the card part: run on the rig (one A10G), from the root of the source tree, after the corpora are in ~/corp
# (tools/bench/real/pack_corpora.sh's layout, the sm_86 binaries built with nvcc 13.0 and -cudart shared).
#   docs/perf-design/data/sp60b/card-run.sh <stage> [out dir]        stages: build ci ab early exit all, and repeat (not part of all)
# Forms of the A/B (tools/bench/hostgpu/run.py): exact and turbo sample launches (the default); exact0 and turbo0 do not
# (VGPU_HOST_GPU_SAMPLE=0: the rule of Round sp59). All four are interleaved, run by run, with the plain simulator.
set -u
stage=${1:?stage}
out=${2:-$HOME/sp60b-out}
root=$(pwd)
build=$HOME/sp60b-build
corp=$HOME/corp
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
LONG="2MM,CORR,COVAR,FDTD-2D,GRAMSCHM,LU"
HEC_SKIP="aidw-cuda,atomicCost-cuda,fpdc-cuda,zerocopy-cuda"

case $stage in
build|all)
  # (what tools/card-ci/run-card-ci.sh configures; it builds the targets the card CI needs, which are the ones the harness needs)
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  ;;&
ci|all)
  # the stand-in-driver test alone, with its output kept: steps 17 to 20 are new and have run nowhere before
  (cd "$build" && ctest -R 'e2e_host_gpu_stub|test_host_gpu_logic|hostgpu_bench_selftest' --output-on-failure) 2>&1 | tee "$out/stub.txt"
  ;;&
ab|all)
  # the A/B: 3 runs a form (1 for a program over 30 s in the simulator), medians. Cut for the rig's hours: GRAMSCHM is left out (229 s in the
  # simulator and more in every card form: 25 minutes), and HeCBench runs are limited to 120 s (the simulator's runs of the programs that take
  # longer are cut there and give no reference time).
  $H --corpora rodinia --runs 3 --long-secs 30 --long-runs 1 --forms sim,exact,exact0,turbo,turbo0 --timeout 600 \
     --json "$out/ab-rodinia.json" --md "$out/ab-rodinia.md" 2>&1 | tee "$out/ab-rodinia.txt" | tail -40
  $H --corpora polybench --skip GRAMSCHM --runs 3 --long-secs 30 --long-runs 1 --forms sim,exact,exact0,turbo,turbo0 --timeout 600 \
     --json "$out/ab-polybench.json" --md "$out/ab-polybench.md" 2>&1 | tee "$out/ab-polybench.txt" | tail -40
  $H --corpora hecbench --skip "$HEC_SKIP" --runs 3 --long-secs 30 --long-runs 1 --forms sim,exact,exact0,turbo,turbo0 --timeout 120 \
     --json "$out/ab-hecbench.json" --md "$out/ab-hecbench.md" 2>&1 | tee "$out/ab-hecbench.txt" | tail -40
  # what each sample measured and what was sent: one run with the log (also gives the forks' cost and the estimates on the rig's own CPU)
  for p in polybench/SYRK polybench/SYR2K polybench/2MM polybench/GEMM polybench/2DCONV polybench/GEMVER rodinia/backprop hecbench/black-scholes-cuda hecbench/binomial-cuda; do
    case $p in
      polybench/*) d=$corp/polybench/bin; cmd="$d/${p#*/}";;
      rodinia/*) d=$(mktemp -d); cmd="$corp/rodinia/bin/backprop 262144";;
      *) d=$corp/hec/src/${p#*/}; cmd="$d/main $(grep "^${p#*/}|" $corp/hec/reduced.txt | cut -d'|' -f2)";;
    esac
    for m in exact turbo; do
      echo "== $p $m"
      ( cd "$d" && env VGPU_QUIET=1 VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$build/shim VGPU_HOST_GPU=$m VGPU_HOST_GPU_LOG=1 timeout 600 $cmd 2>&1 | grep -E "^host_gpu(_sample|_sampled|_turbo|: .*(sent by|on the card|failed))" | cut -c1-600 )
    done
  done > "$out/sample-log.txt" 2>&1
  ;;&
early|all)
  # the driver started early, on its thread (an experiment; the switch was removed after this run, see the round's text): the short and middling
  # programs, where a start that is not used shows
  $H --corpora polybench,rodinia --skip "$LONG" --runs 3 --forms sim,exact,turbo --env VGPU_HOST_GPU_EARLY_START=1 --timeout 300 \
     --json "$out/early.json" --md "$out/early.md" 2>&1 | tee "$out/early.txt" | tail -30
  ;;&
exit|all)
  # whether a process can end while the driver starts (VGPU_HOST_GPU_EARLY_START=2 does not wait at exit; 1 waits): 10 runs each of three short
  # programs, wall seconds and exit status; 0 is the default (the driver is not started early)
  pb=$corp/polybench/bin
  rb=$corp/rodinia/bin
  one() {   # <label> <mode> <early> <cwd> <program and arguments...>
    local label=$1 mode=$2 early=$3 cwd=$4
    shift 4
    for i in $(seq 1 10); do
      ( cd "$cwd" && /usr/bin/time -f "$label $mode early=$early %e s rc=%x" env VGPU_QUIET=1 VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$build/shim VGPU_HOST_GPU=$mode \
          VGPU_HOST_GPU_EARLY_START=$early "$@" > /dev/null ) 2>&1 | tail -1
    done
  }
  for early in 0 1 2; do
    one BICG exact $early "$pb" "$pb/BICG"
    one GESUMMV exact $early "$pb" "$pb/GESUMMV"
    one srad_v2 turbo $early "$(mktemp -d)" "$rb/srad_v2" 512 512 0 127 0 127 0.5 2
  done | tee "$out/exit.txt"
  ;;
repeat)
  # The programs the A/B showed more than 3% apart with NO launch sampled in either form (the two forms then run the same code): 10 more runs of
  # each form, interleaved, to see the spread of one form against the gap between the two. Then what the machine has left of the runs: zombies
  # and processes of the corpus still there.
  $H --programs rodinia/backprop,rodinia/sc_gpu,polybench/JACOBI2D,rodinia/hotspot --runs 10 --forms sim,exact,exact0,turbo,turbo0 --timeout 300 --no-exact-stats \
     --json "$out/repeat.json" --md "$out/repeat.md" 2>&1 | tee "$out/repeat.txt" | tail -12
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the corpus left: $(ps -eo args= | grep -c "^$corp/")"; } | tee "$out/leftover.txt"
  ;;
esac
