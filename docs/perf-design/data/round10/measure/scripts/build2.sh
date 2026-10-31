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

(comp gemm_sm80_sparse_f16 80 gemm/device/gemm_f16t_f16n_f32t_tensor_op_f32_sparse_sm80 ; comp sm90_pingpong 90a gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized_pingpong; comp sm90_sparse_f16 90a gemm/device/sm90_sparse_gemm_f16_f16_f32_tensor_op_f32) &
(comp sm100_f8 100a gemm/device/sm100_tensorop_gemm/f8_f8_void_f32; comp sm100_blockscaled_mxf8 100a gemm/device/sm100_blockscaled_tensorop_gemm/mxf8_mxf6_f16_f8_tn_layout; comp sm120_bs_mxf4 120a gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_mxf4_mxf4_f32_f32) &
wait
echo BUILD2_DONE >> $out/compile_times.txt
