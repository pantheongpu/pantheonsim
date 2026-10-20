#!/bin/bash
# cut.sh <case> <tier> <cpu> <stats> <outprefix> [threads]
case=$1 tier=$2 cpu=$3 stats=$4 out=$5; th=${6:-1}
D=~/sim/m/cut; shim=~/sim/build/shim
export LD_LIBRARY_PATH=$shim VGPU_QUIET=1 VGPU_SASS_LOG=1 VGPU_THREADS=$th
case $case in
 gemm_sm80_sparse_f16) gpu=nvidia/a100; cmd="$D/gemm_sm80_sparse_f16 --gtest_filter=*";;
 sm90_group_gemm) gpu=nvidia/h100; cmd="$D/sm90_group_gemm --gtest_filter=*.128x128x64_2x2x1:*.128x128x64_2x2x1_silu";;
 sm90_pingpong) gpu=nvidia/h100; cmd="$D/sm90_pingpong --gtest_filter=*f16t_f16t_f32n_tensor_op_gmma_f32_persistent.64x128x64_2x4x1:*f16t_f16n_f16t_tensor_op_gmma_f32_persistent_Epilogue.64x128x64_2x2x1";;
 sm90_sparse_f16) gpu=nvidia/h100; cmd="$D/sm90_sparse_f16 --gtest_filter=*";;
 tma_load) gpu=nvidia/h100; cmd="$D/tma_load --gtest_filter=-*Tma_Load_1D";;
 sm100_f16) gpu=nvidia/b200; cmd="$D/sm100_f16_f16_void_f32 --gtest_filter=*-*_1x8x1_streamK";;
 sm100_f8) gpu=nvidia/b200; cmd="$D/sm100_f8 --gtest_filter=*-*streamK";;
 sm100_blockscaled) gpu=nvidia/b200; cmd="$D/sm100_blockscaled_mxf8 --gtest_filter=*4x4x1_1sm_auto";;
 sm120_f4_f6) gpu=nvidia/rtx5090; cmd="$D/sm120_f4_f6 --gtest_filter=*";;
 sm120_bs_mxf4) gpu=nvidia/rtx5090; cmd="$D/sm120_bs_mxf4 --gtest_filter=*";;
 fmha1) gpu=nvidia/b200; cmd="$D/fmha --b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=no --iterations=0";;
 fmha2) gpu=nvidia/b200; cmd="$D/fmha --b=1 --h=4 --q=512 --k=512 --d=128 --verify --mask=causal --iterations=0";;
 fmha3) gpu=nvidia/b200; cmd="$D/fmha --b=2 --h=4 --h_k=2 --q=512 --k=512 --d=64 --verify --iterations=0";;
 fmha4) gpu=nvidia/b200; cmd="$D/fmha --verify --varlen --mask=causal,residual --d=64 --h=4 --h_k=2 --varlen-q=17:10 --varlen-k=13:10 --iterations=0";;
 *) echo unknown; exit 2;;
esac
export VGPU_GPU=$gpu
exec ~/sim/m/run1.sh $out $tier $cpu $stats -- $cmd
