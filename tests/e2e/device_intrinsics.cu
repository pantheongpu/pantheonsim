// Intrinsics and builtins found by compiling representative CUDA and feeding
// the PTX to the parser, then checked here for what they actually compute.
//
// Each of these was refused before the probe found it:
//   atomicInc/atomicDec  -> atom.inc/.dec, which wrap against the operand
//                           rather than counting, so a +1 implementation is
//                           right until the first wrap and wrong after
//   __vabsdiffu4/s4      -> vabsdiff4, per-byte with per-byte sign extension
//   device malloc/free   -> a call to a builtin, previously refused outright
//   h2exp/__habs2/...    -> inline asm that declares ".reg.b16" with no space,
//                           which the lexer saw as one unknown directive
//   assert()             -> __assertfail, which now reports its message and
//                           source location instead of a bare trap
#include <cstdio>
#include <cassert>
#include <cuda_fp16.h>
__global__ void k_incdec(unsigned* out) {
    // atomicInc wraps at the limit, so 10 threads against a limit of 3 cycle
    // 0,1,2,3,0,1,2,3,0,1 -- a plain +1 would just count to 10.
    unsigned v = atomicInc(&out[0], 3u);
    out[1 + threadIdx.x] = v;
}
__global__ void k_dec(unsigned* out) {
    unsigned v = atomicDec(&out[0], 5u);
    out[1 + threadIdx.x] = v;
}
__global__ void k_vabsdiff(unsigned* out, unsigned a, unsigned b) {
    out[0] = __vabsdiffu4(a, b);
    out[1] = __vabsdiffs4(a, b);
}
__global__ void k_malloc(int* out) {
    int* p = (int*)malloc(16 * sizeof(int));
    if (!p) { out[0] = -1; return; }
    for (int i = 0; i < 16; ++i) p[i] = i * 3;
    int s = 0;
    for (int i = 0; i < 16; ++i) s += p[i];
    out[0] = s;
    free(p);
}
__global__ void k_assert(int* out, int n) { assert(n > 100); out[0] = n; }

__global__ void k_h2(float* out, float x) {
    __half2 v = __float2half2_rn(x);
    __half2 e = h2exp(v);
    out[0] = __half2float(__low2half(e));
    __half2 a = __habs2(__hneg2(v));
    out[1] = __half2float(__low2half(a));
}
int main() {
    unsigned* d; cudaMalloc(&d, 64 * sizeof(unsigned));
    unsigned zero = 0; cudaMemcpy(d, &zero, 4, cudaMemcpyHostToDevice);
    k_incdec<<<1, 10>>>(d);
    cudaDeviceSynchronize();
    unsigned h[16]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (unsigned i = 0; i < 10; ++i)
        if (h[1 + i] != i % 4) { std::printf("FAIL inc[%u]: %u want %u\n", i, h[1+i], i % 4); ++bad; }

    unsigned five = 5; cudaMemcpy(d, &five, 4, cudaMemcpyHostToDevice);
    k_dec<<<1, 3>>>(d);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    const unsigned want_dec[3] = {5, 4, 3};
    for (int i = 0; i < 3; ++i)
        if (h[1 + i] != want_dec[i]) { std::printf("FAIL dec[%d]: %u want %u\n", i, h[1+i], want_dec[i]); ++bad; }

    // bytes 0x01,0x80,0xFF,0x10 against 0x05,0x7F,0x01,0x20
    const unsigned A = 0x10FF8001u, B = 0x20017F05u;
    k_vabsdiff<<<1, 1>>>(d, A, B);
    cudaDeviceSynchronize();
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    unsigned refu = 0, refs = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned ab = (A >> (8*i)) & 0xFF, bb = (B >> (8*i)) & 0xFF;
        refu |= (ab > bb ? ab - bb : bb - ab) << (8*i);
        int as = (int)(signed char)ab, bs = (int)(signed char)bb;
        refs |= (unsigned)(as > bs ? as - bs : bs - as) << (8*i);
    }
    if (h[0] != refu) { std::printf("FAIL vabsdiffu4: %08x want %08x\n", h[0], refu); ++bad; }
    if (h[1] != refs) { std::printf("FAIL vabsdiffs4: %08x want %08x\n", h[1], refs); ++bad; }

    int* di; cudaMalloc(&di, 4);
    k_malloc<<<1, 1>>>(di);
    cudaDeviceSynchronize();
    int hi; cudaMemcpy(&hi, di, 4, cudaMemcpyDeviceToHost);
    if (hi != 3 * (15 * 16 / 2)) { std::printf("FAIL device malloc: %d want %d\n", hi, 3*120); ++bad; }

    float* df; cudaMalloc(&df, 8);
    k_h2<<<1, 1>>>(df, -2.0f);
    cudaDeviceSynchronize();
    float hf[2]; cudaMemcpy(hf, df, 8, cudaMemcpyDeviceToHost);
    if (!(hf[0] > 0.13f && hf[0] < 0.14f)) { std::printf("FAIL h2exp(-2): %g\n", hf[0]); ++bad; }
    if (hf[1] != 2.0f) { std::printf("FAIL habs2(hneg2(-2)): %g\n", hf[1]); ++bad; }

    // A failed device assert must report cudaErrorAssert with its message, and
    // the error must survive cudaDeviceSynchronize -- the usual way anyone
    // checks a kernel is sync-then-cudaGetLastError, and a sync that consumed
    // the error turned a kernel that died into one that reported success.
    const cudaError_t before_assert = cudaGetLastError();
    if (before_assert != cudaSuccess) {
        std::printf("cuda error before assert check: %s\n", cudaGetErrorString(before_assert));
        ++bad;
    }
    k_assert<<<1, 1>>>(di, 5);
    cudaDeviceSynchronize();
    const cudaError_t ae = cudaGetLastError();
    if (ae != cudaErrorAssert) {
        std::printf("FAIL assert: got %s want device-side assert triggered\n",
                    cudaGetErrorString(ae));
        ++bad;
    }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
