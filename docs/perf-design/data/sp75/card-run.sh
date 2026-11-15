#!/usr/bin/env bash
# Round sp75, the card part: run on the rig (one A10G), from the root of the source tree, with the corpora in ~/corp
# (tools/bench/real/pack_corpora.sh's layout: the sm_86 binaries built with nvcc 13.0 and -cudart shared, the CUDA Samples under ~/corp/samples).
# Nothing of this was run when the round was written: the round had no card. What each stage should show is in README.md, "The card run".
# Run on 2026-11-15 (card.md), the stages after `build` with SP75_BUILD set to the summary run's Release build.
#   docs/perf-design/data/sp75/card-run.sh <stage> [out dir]        stages: build decisions changed keep small rest, and all (in that order)
# Forms: turbo and exact are this round's rule; turbob0 and exactb0 the same build with VGPU_HOST_GPU_SAMPLE_BLOCK=0, the rule before it (a sample's
# copy runs whole blocks however long, and a grid under 256 blocks with no followed loop is not sampled). The harness interleaves them, run by run.
set -u
stage=${1:?stage: build decisions changed keep small rest all}
out=${2:-$HOME/sp75-out}
build=${SP75_BUILD:-$HOME/sp75-build}
corp=${SP75_CORP:-$HOME/corp}
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH NVCC=/usr/local/cuda-13.0/bin/nvcc
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
S=$corp/samples/Samples
# The programs whose simulator run passes 300 s on the rig, and the seconds their runs get (half an hour of timeouts otherwise).
slow=aligned-types-cuda,babelstream-cuda,base64e-cuda,binomial-cuda
slow_timeout=${SP75_SLOW_TIMEOUT:-90}
# One logged turbo run of a program, cut when it has run `cut` seconds: the mode's lines (the decisions are made before a launch runs).
logged() {   # <tag> <seconds> <cwd> <program and arguments...>
  local tag=$1 secs=$2 cwd=$3
  shift 3
  ( cd "$cwd" && env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 timeout -s KILL "$secs" "$@" > /dev/null 2> "$out/log-$tag.err" )
  grep -E '^host_gpu(_[a-z]+)?: ' "$out/log-$tag.err" | grep -v '^host_gpu:   ' > "$out/log-$tag.txt"
  echo "== $tag"; grep -E 'sampled|starts the driver|host_gpu_nosample|on the card|host_gpu_kept|launches ran' "$out/log-$tag.txt" | cut -c1-400
}

case $stage in
build|all)
  # the build, the unit tests, e2e_host_gpu and e2e_host_gpu_stub (its steps 38 to 40 are this round's and had run nowhere: no nvcc where it was built)
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  nice cmake --build "$build" -j4 --target vgpucuda > "$build.vgpucuda.log" 2>&1 || tail -20 "$build.vgpucuda.log"
  (cd "$build" && ctest -R 'hostgpu_bench_selftest' --output-on-failure) 2>&1 | tail -3 | tee "$out/selftest.txt"
  ;;&
decisions|all)
  # What the mode says of each launch of the round's programs on the card (5 to 40 s each; the long ones are on the card by then or the run is cut).
  logged cudaTensorCoreGemm 60 "$S/3_CUDA_Features/cudaTensorCoreGemm" ./cudaTensorCoreGemm | tee "$out/decisions.txt"
  logged tf32TensorCoreGemm 60 "$S/3_CUDA_Features/tf32TensorCoreGemm" ./tf32TensorCoreGemm | tee -a "$out/decisions.txt"
  logged bf16TensorCoreGemm 60 "$S/3_CUDA_Features/bf16TensorCoreGemm" ./bf16TensorCoreGemm | tee -a "$out/decisions.txt"
  logged immaTensorCoreGemm 60 "$S/3_CUDA_Features/immaTensorCoreGemm" ./immaTensorCoreGemm | tee -a "$out/decisions.txt"
  logged aidw 120 "$corp/hec/src/aidw-cuda" ./main 100 1 1 | tee -a "$out/decisions.txt"
  # #662: what the mode's own work costs the two programs that are slower than the simulator (the host_gpu_overhead line), and which launch
  # buys silu's driver start
  logged silu 60 "$corp/hec/src/silu-cuda" ./main 8 8 64 64 1 | tee -a "$out/decisions.txt"
  logged s3d 60 "$corp/hec/src/s3d-cuda" ./main -q -n 1 -s 1 | tee -a "$out/decisions.txt"
  grep -h 'host_gpu_overhead' "$out/log-silu.txt" "$out/log-s3d.txt" | tee -a "$out/decisions.txt"
  ;;&
changed|all)
  # The programs the rule is for. tf32 and bf16 take 208 s and 159 s in the simulator and under the rule before, aidw 365 s under the rule before
  # and over 600 s in the simulator (its sim run is killed at the timeout and recorded: about 10 minutes of rig time; SP75_NO_AIDW_SIM=1 leaves aidw to
  # the second line, without a sim run). One run a form.
  $H --corpora samples --programs tf32TensorCoreGemm,bf16TensorCoreGemm,cudaTensorCoreGemm,immaTensorCoreGemm --runs 1 --long-secs 5 --long-runs 1 \
     --forms sim,exact,exactb0,turbo,turbob0 --timeout 600 --json "$out/tensor.json" --md "$out/tensor.md" 2>&1 | tee "$out/tensor.txt" | tail -12
  if [[ -z "${SP75_NO_AIDW_SIM:-}" ]]; then
    $H --corpora hecbench --programs aidw-cuda --runs 1 --long-secs 5 --long-runs 1 --forms sim,turbo,turbob0 --timeout 600 \
       --json "$out/aidw.json" --md "$out/aidw.md" 2>&1 | tee "$out/aidw.txt" | tail -8
  else
    for f in turbo turbob0; do
      e=""; [[ $f == turbob0 ]] && e="VGPU_HOST_GPU_SAMPLE_BLOCK=0"
      ( cd "$corp/hec/src/aidw-cuda" && /usr/bin/time -f "aidw $f %e s" env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo $e timeout -s KILL 600 ./main 100 1 1 > "$out/aidw-$f.out" 2> "$out/aidw-$f.err" )
      tail -1 "$out/aidw-$f.err" | tee -a "$out/aidw.txt"
    done
  fi
  # a verify run of the tensor programs (every launch on both sides: what the card's GEMM is against the simulator's). The programs print
  # nothing of their result (the Samples' own check is compiled out), and their TFLOPS line, a timing, makes every card output "different".
  $H --corpora samples --programs cudaTensorCoreGemm --runs 1 --forms sim,verify --timeout 600 --json "$out/tensor-verify.json" --md "$out/tensor-verify.md" 2>&1 | tail -4
  # the other three: the harness wants a sim form, which is killed after a second here (the verify run simulates the launch itself: 3.5 to 210 s)
  $H --corpora samples --programs immaTensorCoreGemm,bf16TensorCoreGemm,tf32TensorCoreGemm --runs 1 --forms sim,verify --timeout 1 --verify-timeout 600 \
     --json "$out/tensor-verify3.json" --md "$out/tensor-verify3.md" 2>&1 | tail -6
  ;;&
keep|all)
  # The keep list of the integration of sp64-sp68 (data/sp69/README.md, run 3) and the other programs with a sampled launch (data/sp75/census.md):
  # their samples are now of bounded blocks. No program more than 3 % slower than its b0 form; outputs as they were.
  $H --corpora polybench,rodinia,hecbench --programs 2MM,3MM,CORR,COVAR,FDTD-2D,LU,SYR2K,SYRK,JACOBI2D,gaussian,lud,particlefilter_float,atomicAggregate-cuda,atomicPerf-cuda,black-scholes-cuda,shmembench-cuda \
     --forms sim,exact,exactb0,turbo,turbob0 --runs 3 --long-secs 20 --json "$out/keep.json" --md "$out/keep.md" 2>&1 | tee "$out/keep.txt" | tail -24
  # binomial apart, its runs killed at $slow_timeout seconds: the simulator and exact mode take over 600 s, turbo 1.4 s
  $H --corpora hecbench --programs binomial-cuda --forms sim,exact,exactb0,turbo,turbob0 --runs 1 --long-secs 20 --timeout "$slow_timeout" \
     --json "$out/keep-binomial.json" --md "$out/keep-binomial.md" 2>&1 | tee "$out/keep-binomial.txt" | tail -6
  ;;&
small|all)
  # The programs of the corpus with a launch on a grid under 256 blocks of a kernel with a loop and no followed loop (census.md): none but the
  # tensor programs counts 30 ms, so none is sampled and each must be at its b0 time. Ten runs: they are short.
  $H --corpora rodinia,hecbench,samples --programs lavaMD,s3d-cuda,lud,radixsort2-cuda,mergeSort,segmentationTreeThrust \
     --forms sim,turbo,turbob0 --runs 10 --long-secs 20 --long-runs 3 --json "$out/small.json" --md "$out/small.md" 2>&1 | tee "$out/small.txt" | tail -12
  ;;&
rest|all)
  # The rest of the corpus, one run a form, turbo against turbob0.
  $H --corpora polybench,rodinia,hecbench,samples --skip tf32TensorCoreGemm,bf16TensorCoreGemm,cudaTensorCoreGemm,immaTensorCoreGemm,aidw-cuda,atomicCost-cuda,fpdc-cuda,zerocopy-cuda,GRAMSCHM,$slow \
     --runs 1 --forms sim,turbo,turbob0 --timeout 300 --json "$out/rest.json" --md "$out/rest.md" 2>&1 | tee "$out/rest.txt" | tail -30
  # the four slow programs (turbo: 1.4 to 13 s) apart, their runs killed at $slow_timeout seconds
  $H --corpora hecbench --programs "$slow" --runs 1 --forms sim,turbo,turbob0 --timeout "$slow_timeout" \
     --json "$out/rest-slow.json" --md "$out/rest-slow.md" 2>&1 | tee "$out/rest-slow.txt" | tail -8
  { echo "zombies: $(ps -eo stat= | grep -c '^Z')"; echo "processes of the corpus left: $(ps -eo args= | grep -c "^$corp/")"; } | tee "$out/leftover.txt"
  ;;
esac
