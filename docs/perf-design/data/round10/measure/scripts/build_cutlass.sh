#!/bin/bash
# runs inside drx (cwd /w). Builds CUTLASS tests with SASS (sm_XXa + compute_XXa)
build=/w/build
need_gtest=1
. /w/nvidia/tests/e2e/cutlass_fetch.sh
u="$cutlass/test/unit"
out=/w/m/cut
comp() { # name arch srcrel
  local name=$1 arch=$2 src=$3 s=$SECONDS
  nvcc -std=c++17 -O1 -cudart shared -arch=compute_$arch -code=sm_$arch,compute_$arch --expt-relaxed-constexpr \
    -DCUTLASS_TARGET_NAME="\"$name\"" -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$u/common" -I "$u" -I "$cutlass/test" \
    -I "$gtest/googletest/include" "$u/$src.cu" "$u/test_unit.cpp" "$u/common/filter_architecture.cpp" "$gtest/libgtest.a" -o "$out/$name" >"$out/$name.log" 2>&1
  echo "$name rc=$? compile_s=$((SECONDS-s))" >> $out/compile_times.txt
}
fmha() {
  local s=$SECONDS ex="$cutlass/examples/77_blackwell_fmha"
  nvcc -std=c++17 -O1 -cudart shared -arch=compute_100a -code=sm_100a,compute_100a --expt-relaxed-constexpr --use_fast_math \
   -I "$cutlass/include" -I "$cutlass/tools/util/include" -I "$cutlass/examples/common" -I "$ex" "$ex/77_blackwell_fmha.cu" -o $out/fmha >$out/fmha.log 2>&1
  echo "fmha rc=$? compile_s=$((SECONDS-s))" >> $out/compile_times.txt
}
echo "fetched; cutlass=$cutlass"
# two at a time (each compile peaks near 10 GB)
(comp gemm_sm80_sparse_f16 80a gemm/device/gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80 ; comp sm90_group_gemm 90a gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_group_gemm; comp tma_load 90a cute/hopper/tma_load) &
(fmha; comp sm100_f16_f16_void_f32 100a gemm/device/sm100_tensorop_gemm/f16_f16_void_f32; comp sm120_f4_f6 120a gemm/device/sm120_tensorop_gemm/sm120_gemm_f4_f6_f32_tensor_op) &
wait
echo BUILD_DONE >> $out/compile_times.txt
