#!/usr/bin/env bash
# Builds the FTTN programs (tools/fttn/fetch.sh <dir> first) and the motivating example.
#
#   tools/fttn/build.sh nvidia <dir> [sm ...]      nvcc, one set a compute architecture (default: 75 80 86 89 90 100 120; CUDA 13 has
#                                                  no sm_70) into <dir>/bin/sm_<N>/{fp16in,bf16in,tf-fp32in,fp64in,motivating}
#                                                  (sm_75: fp16in and motivating only; it has no bf16, tf32 or f64 form)
#   tools/fttn/build.sh amd <dir> [gfx ...]        hipcc (ROCM_PATH), default gfx90a gfx942 gfx1100, into <dir>/bin-amd/<gfx>/
#
# FTTN_TILES=512 builds the motivating example at the paper's full size (8192 x 8192 outputs): for a card, not for the simulator.
set -uo pipefail
what="${1:?usage: $0 nvidia|amd <dir> [arch ...]}"; dir="${2:?}"; shift 2
src="$dir/FTTN"; tiles="${FTTN_TILES:-1}"
cd "$src" || exit 1
if [[ $what == nvidia ]]; then
  nvcc_bin="${NVCC:-nvcc}"
  for a in ${*:-75 80 86 89 90 100 120}; do
    mkdir -p "$dir/bin/sm_$a"
    only=(); tests=(fp16in bf16in tf-fp32in fp64in)
    if (( a < 80 )); then only=(-DFTTN_FP16_ONLY); tests=(fp16in); fi
    for t in "${tests[@]}"; do
      "$nvcc_bin" --std=c++17 -I./utility/ -DNVIDIA "${only[@]}" -arch="sm_$a" -cudart shared -w -x cu "$t.cpp" -o "$dir/bin/sm_$a/$t" || echo "FAIL sm_$a $t"
    done
    "$nvcc_bin" --std=c++17 -I. -I./utility/ -DNVIDIA "${only[@]}" -DM_TILES="$tiles" -DN_TILES="$tiles" -arch="sm_$a" -cudart shared -w -x cu motivating.cpp \
      -o "$dir/bin/sm_$a/motivating" || echo "FAIL sm_$a motivating"
  done
else
  rocm="${ROCM_PATH:-/opt/rocm}"
  for a in ${*:-gfx90a gfx942 gfx1100}; do
    mkdir -p "$dir/bin-amd/$a"
    for t in fp16in bf16in tf-fp32in fp64in; do
      "$rocm/bin/hipcc" --std=c++17 -I./utility/ -DAMD -I"$dir/rocWMMA/library/include" --offload-arch="$a" -w "$t.cpp" -o "$dir/bin-amd/$a/$t" || echo "FAIL $a $t"
    done
    "$rocm/bin/hipcc" --std=c++17 -I. -I./utility/ -DAMD -DM_TILES="$tiles" -DN_TILES="$tiles" -I"$dir/rocWMMA/library/include" --offload-arch="$a" -w motivating.cpp \
      -o "$dir/bin-amd/$a/motivating" || echo "FAIL $a motivating"
  done
fi
