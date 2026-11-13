#!/usr/bin/env bash
# Round sp65, the card part: run on the rig (one A10G), from the root of the source tree, after the corpora are in ~/corp
# (tools/bench/real/pack_corpora.sh's layout, the sm_86 binaries built with nvcc 13.0 and -cudart shared). Built on sp61's card-run.sh.
#   docs/perf-design/data/sp65/card-run.sh <stage> [out dir]        stages: build ci target ab log all, and repeat (not part of all)
# Forms: exact and turbo send a launch to the card while the card holds the live memory when the pull it saves repays it (the default of this
# round); exactn and turbon do not (VGPU_HOST_GPU_STREAK=0: the rule of Round sp61, the bar's reference). Interleaved, run by run, with the plain
# simulator. exactdiv (exact with VGPU_HOST_GPU_DIV=1) is in the target stage because it admits LU's and gaussian's division kernels, which
# the streak rule then reaches in exact mode too; its reference is a run of exactdiv with VGPU_HOST_GPU_STREAK=0, the stage's second pass.
set -u
stage=${1:?stage}
out=${2:-$HOME/sp65-out}
root=$(pwd)
build=$HOME/sp65-build
corp=$HOME/corp
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
HEC_SKIP="aidw-cuda,atomicCost-cuda,fpdc-cuda,zerocopy-cuda"
# the programs whose decisions the offline replay says the rule changes (offline/changes-turbo.md: LU, gaussian, lud), two with launches kept
# for their size that it says it does not change (sc_gpu, hotspot), and GRAMSCHM apart (about 4 minutes a run)
TARGET=${SP65_TARGET:-polybench/LU,rodinia/gaussian,rodinia/lud,rodinia/sc_gpu,rodinia/hotspot}
GRAMSCHM=polybench/GRAMSCHM

case $stage in
build|all)
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  ;;&
ci|all)
  # the stand-in-driver test (steps 24 to 26 are new and have run nowhere), the logic test and the harness's self-test
  (cd "$build" && ctest -R 'e2e_host_gpu_stub|test_host_gpu_logic|hostgpu_bench_selftest' --output-on-failure) 2>&1 | tee "$out/stub.txt"
  ;;&
target|all)
  # the programs the rule changes: 3 runs of each form (LU: about 30 s a run in the simulator, so --long-secs 60 keeps its 3 runs)
  $H --programs "$TARGET" --runs 3 --long-secs 60 --long-runs 1 --forms sim,exact,exactn,turbo,turbon,exactdiv --timeout 600 \
     --json "$out/target.json" --md "$out/target.md" 2>&1 | tee "$out/target.txt" | tail -20
  $H --programs "$TARGET" --runs 3 --long-secs 60 --long-runs 1 --forms sim,exactdiv --env VGPU_HOST_GPU_STREAK=0 --timeout 600 --no-exact-stats \
     --json "$out/target-div0.json" --md "$out/target-div0.md" 2>&1 | tee "$out/target-div0.txt" | tail -12
  # GRAMSCHM: one run a form (230 s in the simulator, mostly its CPU reference); no extra logged exact run (the log stage has its lines)
  $H --programs "$GRAMSCHM" --runs 1 --forms sim,exact,exactn,turbo,turbon --timeout 900 --no-exact-stats \
     --json "$out/gramschm.json" --md "$out/gramschm.md" 2>&1 | tee "$out/gramschm.txt" | tail -12
  ;;&
ab|all)
  # the bar for everything else: one run a form; their decisions are the old rule's by the replay (no launch of theirs is sent by the new rule)
  $H --corpora polybench,rodinia --skip "GRAMSCHM,2MM,CORR,COVAR,LU,gaussian,lud,sc_gpu,hotspot" --runs 1 --forms sim,exact,exactn,turbo,turbon --timeout 300 \
     --json "$out/ab-rest.json" --md "$out/ab-rest.md" 2>&1 | tee "$out/ab-rest.txt" | tail -30
  $H --corpora hecbench --skip "$HEC_SKIP,aligned-types-cuda,atomicReduction-cuda,babelstream-cuda,base64e-cuda,binomial-cuda" --runs 1 \
     --forms sim,exact,exactn,turbo,turbon --timeout 120 --json "$out/ab-hec.json" --md "$out/ab-hec.md" 2>&1 | tee "$out/ab-hec.txt" | tail -30
  ;;&
log|all)
  # question 2 on the card: every launch's kernel time on the card against the simulator's estimate, and the launches the rule sent
  for p in polybench/bin/GRAMSCHM polybench/bin/LU; do
    for m in turbo turbon; do
      env="VGPU_HOST_GPU=turbo"; [[ $m == turbon ]] && env="VGPU_HOST_GPU=turbo VGPU_HOST_GPU_STREAK=0"
      echo "== $p $m"
      ( cd "$corp/$(dirname $p)" && env VGPU_QUIET=1 VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$build/shim $env VGPU_HOST_GPU_LOG=1 timeout 900 ./$(basename $p) 2>&1 \
        | python3 "$root/docs/perf-design/data/sp65/cardlog.py" )
    done
  done > "$out/card-log.txt" 2>&1
  for a in "rodinia/bin/gaussian -s 512" "rodinia/bin/lud -s 1024 -v"; do
    set -- $a
    for m in turbo turbon; do
      env="VGPU_HOST_GPU=turbo"; [[ $m == turbon ]] && env="VGPU_HOST_GPU=turbo VGPU_HOST_GPU_STREAK=0"
      echo "== $1 $m"
      d=$(mktemp -d); ( cd "$d" && env VGPU_QUIET=1 VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$build/shim $env VGPU_HOST_GPU_LOG=1 timeout 600 "$corp/$1" "${@:2}" 2>&1 \
        | python3 "$root/docs/perf-design/data/sp65/cardlog.py" ); rm -rf "$d"
    done
  done >> "$out/card-log.txt" 2>&1
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the corpus left: $(ps -eo args= | grep -c "^$corp/")"; } | tee "$out/leftover.txt"
  ;;
repeat)
  # ten more runs of each form of any program the A/B shows more than 3% apart
  $H --programs "${SP65_REPEAT:?programs}" --runs 10 --forms sim,exact,exactn,turbo,turbon --timeout 300 --no-exact-stats \
     --json "$out/repeat.json" --md "$out/repeat.md" 2>&1 | tee "$out/repeat.txt" | tail -12
  ;;
esac
