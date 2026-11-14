#!/usr/bin/env bash
# Round sp66, the card part: run on the rig (one A10G), from the root of the source tree, after the corpora are in ~/corp
# (tools/bench/real/pack_corpora.sh's layout, the sm_86 binaries built with nvcc 13.0 and -cudart shared; the CUDA Samples under ~/corp/samples as
# tools/bench/real/real.py's manifest reads them, if they are wanted). Built on sp61's card-run.sh.
#   docs/perf-design/data/sp66/card-run.sh <stage> [out dir]        stages: build changed atomics rest, and all (the four in that order)
# Forms: exact and turbo are this round's rules; exactw0 and turbow0 the same build with VGPU_HOST_GPU_WIDER=0, the safety rules before the round
# (named exactn and turbon when the round ran; renamed when it was merged with sp65, whose exactn and turbon are VGPU_HOST_GPU_STREAK=0)
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
# The programs whose decisions change (data/sp66/census.md: a launch kept by a rule this round lifts, in the old census); the CUDA Samples among them
# run only when ~/corp/samples holds them (sim_main.txt and the sm_86 binaries, the layout of tools/bench/real/real.py's manifest).
CHANGED=${SP66_CHANGED:-hecbench/aop-cuda,hecbench/asyncAllocation-cuda,hecbench/atomicPerf-cuda,hecbench/attention-cuda,hecbench/bgmv-cuda,hecbench/bh-cuda,hecbench/radixsort2-cuda,hecbench/sortKV-cuda,rodinia/particlefilter_float,samples/LargeKernelParameter,samples/MC_EstimatePiInlineP,samples/MC_SingleAsianOptionP,samples/bf16TensorCoreGemm,samples/binaryPartitionCG,samples/cdpBezierTessellation,samples/cdpQuadtree,samples/cdpSimplePrint,samples/cdpSimpleQuicksort,samples/convolutionSeparable,samples/cudaGraphsPerfScaling,samples/cudaTensorCoreGemm,samples/dct8x8,samples/histogram,samples/interval,samples/jacobiCudaGraphs,samples/mergeSort,samples/newdelete,samples/ptxjit,samples/radixSortThrust,samples/reduction,samples/scalarProd,samples/segmentationTreeThrust,samples/shfl_scan,samples/simpleAtomicIntrinsics,samples/simpleCooperativeGroups,samples/simpleCubemapTexture,samples/simpleCudaGraphs,samples/simpleHyperQ,samples/simpleLayeredTexture,samples/simplePrintf,samples/simpleSurfaceWrite,samples/sortingNetworks,samples/tf32TensorCoreGemm,samples/threadFenceReduction,samples/warpAggregatedAtomicsCG}
CORPORA=${SP66_CORPORA:-polybench,rodinia,hecbench}
CHANGED_CORPORA=$CORPORA
[[ -f $corp/samples/sim_main.txt ]] && CHANGED_CORPORA=$CORPORA,samples   # (the CUDA Samples, when they were copied: census.md has some of them)

case $stage in
build|all)
  # builds, the unit tests, e2e_host_gpu (steps 7 to 9 are new: module variables, printf and a texture object on the card) and e2e_host_gpu_stub
  # (steps 24 to 28 when the round ran, 29 to 35 since the merge with sp65 and sp67: new, and they had run nowhere: no nvcc on the machine the
  # round was built on), the probes
  CARD_CI_JOBS=4 tools/card-ci/run-card-ci.sh "$build" 2>&1 | tee "$out/card-ci.txt"
  nice cmake --build "$build" -j4 --target vgpucuda > "$build.vgpucuda.log" 2>&1 || tail -20 "$build.vgpucuda.log"   # (the driver shim: Samples that call the driver API)
  (cd "$build" && ctest -R 'hostgpu_bench_selftest' --output-on-failure) 2>&1 | tail -3 | tee "$out/selftest.txt"
  ;;&
changed|all)
  # 3 runs a form; one untimed verify run (VGPU_HOST_GPU_VERIFY=all) of each
  $H --corpora "$CHANGED_CORPORA" --programs "$CHANGED" --runs 3 --long-secs 20 --long-runs 1 --forms sim,exact,exactw0,turbo,turbow0,verify --timeout 600 \
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
