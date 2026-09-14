// A full 16x16x16 matmul through the public WMMA API: load_matrix_sync on both
// operands and the accumulator, mma_sync, store_matrix_sync. The fragment
// layout is opaque, so this checks the only thing that is observable -- the
// product -- against a host reference.
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Row-major B, and the col_major variant below: a real GEMM uses the second,
// and the two layouts cancel differently through the fragment, so passing one
// is no evidence for the other.
__global__ void gemm(float* d, const half* a, const half* b, const float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::load_matrix_sync(acc, c, 16, wmma::mem_row_major);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

__global__ void gemm_colb(float* d, const half* a, const half* b, const float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::load_matrix_sync(acc, c, 16, wmma::mem_row_major);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

// bf16 fragments are a different shape from f16 -- 4 registers instead of 8,
// covering the matrix once instead of twice -- so this is a separate path
// through both load and mma, not the same one with another decode.
__global__ void gemm_bf16(float* d, const __nv_bfloat16* a, const __nv_bfloat16* b,
                          const float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::load_matrix_sync(fa, a, 16);
    wmma::load_matrix_sync(fb, b, 16);
    wmma::load_matrix_sync(acc, c, 16, wmma::mem_row_major);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

// tf32's m16n16k8 is a different shape entirely: A is 16x8 and B is 8x16, so
// A and B do not share an index map and the layout cancellation the square
// shapes rely on does not hold. Its load computes the address from the layout
// directly, which is a separate code path worth its own answer.
__global__ void gemm_tf32(float* d, const float* a, const float* b, const float* c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::col_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
    wmma::load_matrix_sync(fa, a, 8);
    wmma::load_matrix_sync(fb, b, 8);
    wmma::load_matrix_sync(acc, c, 16, wmma::mem_row_major);
    wmma::mma_sync(acc, fa, fb, acc);
    wmma::store_matrix_sync(d, acc, 16, wmma::mem_row_major);
}

int main() {
    half ha[256], hb[256];
    float hc[256], hd[256], ref[256];
    for (int i = 0; i < 256; ++i) {
        ha[i] = __float2half(static_cast<float>((i % 7) - 3));
        hb[i] = __float2half(static_cast<float>((i % 5) - 2));
        hc[i] = static_cast<float>(i % 3);
    }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float acc = hc[i * 16 + j];
            for (int k = 0; k < 16; ++k)
                acc += __half2float(ha[i * 16 + k]) * __half2float(hb[k * 16 + j]);
            ref[i * 16 + j] = acc;
        }
    half *da, *db; float *dc, *dd;
    cudaMalloc(&da, sizeof ha); cudaMalloc(&db, sizeof hb);
    cudaMalloc(&dc, sizeof hc); cudaMalloc(&dd, sizeof hd);
    cudaMemcpy(da, ha, sizeof ha, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof hb, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, hc, sizeof hc, cudaMemcpyHostToDevice);
    gemm<<<1, 32>>>(dd, da, db, dc);
    cudaDeviceSynchronize();
    cudaMemcpy(hd, dd, sizeof hd, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < 256; ++i)
        if (hd[i] != ref[i]) {
            if (bad < 3) std::printf("  row-major B [%d,%d] got %g want %g\n",
                                     i / 16, i % 16, hd[i], ref[i]);
            ++bad;
        }
    // Same product with B held column-major, which is what a GEMM actually
    // does. The reference transposes B rather than the answer changing.
    float refc[256];
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float acc = hc[i * 16 + j];
            for (int k = 0; k < 16; ++k)
                acc += __half2float(ha[i * 16 + k]) * __half2float(hb[j * 16 + k]);
            refc[i * 16 + j] = acc;
        }
    gemm_colb<<<1, 32>>>(dd, da, db, dc);
    cudaDeviceSynchronize();
    cudaMemcpy(hd, dd, sizeof hd, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; ++i)
        if (hd[i] != refc[i]) {
            if (bad < 6) std::printf("  col-major B [%d,%d] got %g want %g\n",
                                     i / 16, i % 16, hd[i], refc[i]);
            ++bad;
        }
    // The same column-major product again, in bf16. The values are small
    // integers, exact in both formats, so the reference does not change.
    __nv_bfloat16 hab[256], hbb[256];
    for (int i = 0; i < 256; ++i) {
        hab[i] = __float2bfloat16(static_cast<float>((i % 7) - 3));
        hbb[i] = __float2bfloat16(static_cast<float>((i % 5) - 2));
    }
    __nv_bfloat16 *dab, *dbb;
    cudaMalloc(&dab, sizeof hab); cudaMalloc(&dbb, sizeof hbb);
    cudaMemcpy(dab, hab, sizeof hab, cudaMemcpyHostToDevice);
    cudaMemcpy(dbb, hbb, sizeof hbb, cudaMemcpyHostToDevice);
    gemm_bf16<<<1, 32>>>(dd, dab, dbb, dc);
    cudaDeviceSynchronize();
    cudaMemcpy(hd, dd, sizeof hd, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; ++i)
        if (hd[i] != refc[i]) {
            if (bad < 9) std::printf("  bf16 [%d,%d] got %g want %g\n",
                                     i / 16, i % 16, hd[i], refc[i]);
            ++bad;
        }

    // tf32, k = 8. Small integers again, so tf32's 10-bit mantissa is exact
    // and the reference is a plain integer product.
    float ta[128], tb[128], tref[256];
    for (int i = 0; i < 128; ++i) { ta[i] = float((i % 7) - 3); tb[i] = float((i % 5) - 2); }
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float acc = hc[i * 16 + j];
            for (int k = 0; k < 8; ++k) acc += ta[i * 8 + k] * tb[j * 8 + k];
            tref[i * 16 + j] = acc;
        }
    float *dta, *dtb;
    cudaMalloc(&dta, sizeof ta); cudaMalloc(&dtb, sizeof tb);
    cudaMemcpy(dta, ta, sizeof ta, cudaMemcpyHostToDevice);
    cudaMemcpy(dtb, tb, sizeof tb, cudaMemcpyHostToDevice);
    gemm_tf32<<<1, 32>>>(dd, dta, dtb, dc);
    cudaDeviceSynchronize();
    cudaMemcpy(hd, dd, sizeof hd, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; ++i)
        if (hd[i] != tref[i]) {
            if (bad < 12) std::printf("  tf32 [%d,%d] got %g want %g\n",
                                      i / 16, i % 16, hd[i], tref[i]);
            ++bad;
        }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; }
    std::printf(bad ? "FAILED (%d of 1024 wrong)\n" : "PASS\n", bad);
    return bad ? 1 : 0;
}
