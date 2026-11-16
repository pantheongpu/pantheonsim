#!/usr/bin/env bash
# #661, the one part of it a card decides: does BMMA (1-bit tensor MMA, sm_75+/sm_80) give the simulator's bits on the card? Run on the rig (one A10G),
# from the root of the tree, corpora in ~/corp, build in SP76_BUILD (as card-run.sh). Not run when written (no card on the machine that wrote it).
#   docs/perf-design/data/turbo-left/card-bmma-verify.sh [out dir]
# BMMA computes integer popcounts, so the card must equal the simulator byte for byte: verify=all in EXACT mode compares bytes per 64 KiB chunk
# and also runs the kernels the exact set refuses ("other_*" in the host_gpu_verify line). Read out/bmma-exact.txt:
#   other_different=0 and other_identical>0   -> BMMA matches on this card: it can join the tensor class (include/vgpu/host_gpu.hpp, turbo_class) with a unit test, and
#                                                the stand-in test's tensor step; then rerun with VGPU_HOST_GPU=turbo to see the speed.
#   other_different>0                         -> do not admit it; the first differing chunk's address is in out/bmma-turbo-verify.txt (words_outside).
# If bmmaTensorCoreGemm is not in the corpus, build it from the CUDA 13.0 samples (Samples/3_CUDA_Features/bmmaTensorCoreGemm, -arch=sm_86) and set SP76_BMMA_EXE.
set -u
out=${1:-$HOME/turboleft-out}
build=${SP76_BUILD:-$HOME/sp76-build}
corp=${SP76_CORP:-$HOME/corp}
mkdir -p "$out"
exe=${SP76_BMMA_EXE:-$(find "$corp/samples" -type f -name bmmaTensorCoreGemm -perm -u+x | head -1)}
[[ -n "$exe" ]] || { echo "no bmmaTensorCoreGemm: set SP76_BMMA_EXE"; exit 1; }
run() {   # <tag> <env...>
  local tag=$1; shift
  ( cd "$(dirname "$exe")" && env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU_LOG=1 "$@" timeout -s KILL 900 "$exe" > "$out/$tag.out" 2> "$out/$tag.err"; echo "rc=$?" >> "$out/$tag.err" )
  grep -E '^host_gpu(_[a-z]+)?: ' "$out/$tag.err" | grep -v '^host_gpu:   ' > "$out/$tag.txt"
}
run bmma-sim VGPU_HOST_GPU=off
run bmma-exact VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=all
run bmma-turbo-verify VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=all VGPU_HOST_GPU_ATOL=0 VGPU_HOST_GPU_RTOL=0
run bmma-turbo VGPU_HOST_GPU=turbo
for t in bmma-exact bmma-turbo-verify; do echo "== $t"; grep -E 'host_gpu_verify|host_gpu_turbo|host_gpu_outside|host_gpu_refused|BMMA' "$out/$t.txt" | cut -c1-300; done | tee "$out/bmma-exact.txt"
cmp -s "$out/bmma-sim.out" "$out/bmma-turbo.out" && echo "turbo output: the same as the simulator's" | tee -a "$out/bmma-exact.txt" || echo "turbo output: DIFFERENT" | tee -a "$out/bmma-exact.txt"
grep -E 'launches ran|kept|refus' "$out/bmma-turbo.txt" | head -5 | cut -c1-250
