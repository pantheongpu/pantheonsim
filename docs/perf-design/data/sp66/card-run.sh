#!/usr/bin/env bash
# Round sp66, the card part: run on the rig (one A10G), from the root of the source tree, after the corpora are in ~/corp
# (tools/bench/real/pack_corpora.sh's layout, the sm_86 binaries built with nvcc 13.0 and -cudart shared; the CUDA Samples under ~/corp/samples as
# tools/bench/real/real.py's manifest reads them, if they are wanted). Built on sp61's card-run.sh.
#   docs/perf-design/data/sp66/card-run.sh <stage> [out dir]        stages: build changed atomics rest, and all (the four in that order)
# Forms: exact and turbo are this round's rules; exactn and turbon the same build with VGPU_HOST_GPU_WIDER=0, the safety rules before the round
# (a module's variables, external functions or device-side launches refuse all its kernels, a live texture object every launch, and turbo mode's
# float atomics with a value, CAS and EXCH are an opt-in). Interleaved, run by run, with the plain simulator.
set -u
stage=${1:?stage}
out=${2:-$HOME/sp66-out}
root=$(pwd)
build=$HOME/sp66-build
corp=$HOME/corp
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
HEC_SKIP="aidw-cuda,atomicCost-cuda,fpdc-cuda,zerocopy-cuda"
# The programs whose decisions change (data/sp66/census.md: a launch kept by a rule this round lifts, in the census of either rule).
CHANGED=${SP66_CHANGED:?the programs of census.md, comma separated}
CORPORA=${SP66_CORPORA:-polybench,rodinia,hecbench}
CHANGED_CORPORA=$CORPORA
[[ -f $corp/samples/sim_main.txt ]] && CHANGED_CORPORA=$CORPORA,samples   # (the CUDA Samples, when they were copied: census.md has some of them)

case $stage in
build|all)
  # builds, the unit tests, e2e_host_gpu (steps 7 to 9 are new: module variables, printf and a texture object on the card) and e2e_host_gpu_stub
  # (steps 24 to 28 are new and have run nowhere: no nvcc on the machine the round was built on), the probes
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  (cd "$build" && ctest -R 'hostgpu_bench_selftest' --output-on-failure) 2>&1 | tail -3 | tee "$out/selftest.txt"
  ;;&
changed|all)
  # 3 runs a form; one untimed verify run (VGPU_HOST_GPU_VERIFY=all) of each
  $H --corpora "$CHANGED_CORPORA" --programs "$CHANGED" --runs 3 --long-secs 60 --long-runs 1 --forms sim,exact,exactn,turbo,turbon,verify --timeout 600 \
     --json "$out/changed.json" --md "$out/changed.md" 2>&1 | tee "$out/changed.txt" | tail -40
  ;;&
atomics|all)
  # The hang risk of compare-and-swap loops on the card: per kernel with a float atomic with a value, a CAS or an EXCH, its kernel time a launch on the
  # card against the simulator's, both from one VGPU_HOST_GPU_VERIFY=all run; then a plain run with the defaults (data/sp66/atomics.py).
  python3 docs/perf-design/data/sp66/atomics.py --build "$build" --corp "$corp" --out "$out/atomics" --programs "${SP66_ATOMICS:-aop-cuda,atomicPerf-cuda,bh-cuda}" | tee "$out/atomics.md"
  ;;&
rest|all)
  # the rest of the corpus, one run a form: their decisions are the old rule's by construction (no launch of theirs is kept by a rule this round lifts)
  $H --corpora "$CORPORA" --skip "$CHANGED,$HEC_SKIP,GRAMSCHM,2MM,LU,FDTD-2D" --runs 1 --forms sim,exact,turbo --timeout 300 \
     --json "$out/rest.json" --md "$out/rest.md" 2>&1 | tee "$out/rest.txt" | tail -30
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the corpus left: $(ps -eo args= | grep -c "^$corp/")"; } | tee "$out/leftover.txt"
  ;;
esac
