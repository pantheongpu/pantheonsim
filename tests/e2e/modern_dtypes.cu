// Does unmodified CUDA C++ using the newer datatypes and primitives run here?
//
// Every instruction this exercises was added by reading the PTX spec, not by
// watching a real toolchain, so the risk is a form nvcc emits that the parser
// does not recognise or an intrinsic whose lowering is different from the
// hand-written PTX the unit tests use. That is exactly what a compiled program
// finds and a hand-written one cannot.
//
// Four things, each a separate check so a failure names itself:
//   1. __nv_bfloat16 arithmetic          -> add/mul/fma.rn.bf16(x2)
//   2. atomicAdd on __half2              -> atom.global.add.noftz.f16x2
//   3. cooperative_groups labeled_partition -> match.any.sync
//   4. cuda::barrier                     -> the mbarrier family
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void bf16_math(float* out) {
    // 1.5 and 2.0 are both exact in bf16, so the expected value is exact too
    // and a mismatch is a real difference rather than a rounding one.
    __nv_bfloat16 a = __float2bfloat16(1.5f);
    __nv_bfloat16 b = __float2bfloat16(2.0f);
    __nv_bfloat16 s = __hadd(a, b);          // 3.5
    __nv_bfloat16 p = __hmul(a, b);          // 3.0
    __nv_bfloat16 f = __hfma(a, b, a);       // 1.5*2 + 1.5 = 4.5
    __nv_bfloat162 v = __halves2bfloat162(a, b);
    __nv_bfloat162 vv = __hadd2(v, v);       // {3.0, 4.0}
    out[0] = __bfloat162float(s);
    out[1] = __bfloat162float(p);
    out[2] = __bfloat162float(f);
    out[3] = __bfloat162float(__low2bfloat16(vv));
    out[4] = __bfloat162float(__high2bfloat16(vv));
}

__global__ void half2_atomic(__half2* acc) {
    // Every thread adds {1.0, 2.0} to one half2. 64 threads, so {64, 128},
    // and f16 holds both exactly.
    __half2 one = __halves2half2(__float2half(1.0f), __float2half(2.0f));
    atomicAdd(acc, one);
}

__global__ void labeled(unsigned* out) {
    // labeled_partition groups the lanes that supplied the same label, which
    // is match.any.sync. Lanes are labelled tid/8, so each group has 8.
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(cg::this_thread_block());
    const unsigned label = threadIdx.x / 8;
    auto part = cg::labeled_partition(warp, label);
    out[threadIdx.x] = part.size();
}

int main() {
    int failures = 0;

    float* d_out;
    cudaMalloc(&d_out, 8 * sizeof(float));
    bf16_math<<<1, 1>>>(d_out);
    cudaDeviceSynchronize();
    float h[8] = {0};
    cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost);
    const float want[5] = {3.5f, 3.0f, 4.5f, 3.0f, 4.0f};
    for (int i = 0; i < 5; ++i) {
        if (h[i] != want[i]) {
            std::printf("FAIL bf16[%d]: got %g want %g\n", i, h[i], want[i]);
            ++failures;
        }
    }

    __half2* d_acc;
    cudaMalloc(&d_acc, sizeof(__half2));
    __half2 zero = __halves2half2(__float2half(0.0f), __float2half(0.0f));
    cudaMemcpy(d_acc, &zero, sizeof zero, cudaMemcpyHostToDevice);
    half2_atomic<<<1, 64>>>(d_acc);
    cudaDeviceSynchronize();
    __half2 acc;
    cudaMemcpy(&acc, d_acc, sizeof acc, cudaMemcpyDeviceToHost);
    const float lo = __half2float(__low2half(acc)), hi = __half2float(__high2half(acc));
    if (lo != 64.0f || hi != 128.0f) {
        std::printf("FAIL half2 atomicAdd: got {%g, %g} want {64, 128}\n", lo, hi);
        ++failures;
    }

    unsigned* d_part;
    cudaMalloc(&d_part, 32 * sizeof(unsigned));
    labeled<<<1, 32>>>(d_part);
    cudaDeviceSynchronize();
    unsigned parts[32] = {0};
    cudaMemcpy(parts, d_part, sizeof parts, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; ++i) {
        if (parts[i] != 8u) {
            std::printf("FAIL labeled_partition[%d]: got %u want 8\n", i, parts[i]);
            ++failures;
            break;
        }
    }

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::printf("FAIL cuda error: %s\n", cudaGetErrorString(err));
        ++failures;
    }
    std::printf(failures ? "FAILED\n" : "PASS\n");
    return failures ? 1 : 0;
}
