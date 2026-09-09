// Non-inlined device functions: a real call, not an inlined body.
//
// nvcc inlines aggressively, so this needs __noinline__ to force the shape --
// but the same shape comes out of separate compilation (-rdc=true), of any
// function too large to inline, and of recursion, which cannot be inlined at
// all. Everything here was refused before: the parser stopped at ".func".
//
// Three things worth having separately:
//   slow_add / slow_scale  a plain call, integer and float, with a return
//   branchy                divergence *inside* the callee, which has its own
//                          path stack while the caller's is set aside
//   fact                   recursion, which is what proves the frame is real
//                          rather than an inlining trick
#include <cstdio>
__device__ __noinline__ int slow_add(int a, int b) { return a + b; }
__device__ __noinline__ float slow_scale(float x, float s) { return x * s + 1.0f; }
__device__ __noinline__ int branchy(int n) {   // divergence inside the callee
    if (n % 2 == 0) return n * 10;
    if (n % 3 == 0) return n * 100;
    return n;
}
__device__ __noinline__ int fact(int n) { return n <= 1 ? 1 : n * fact(n - 1); }

__global__ void k(int* out, float* fout) {
    const int t = threadIdx.x;
    out[t] = slow_add(t, 7);
    fout[t] = slow_scale((float)t, 2.0f);
    out[32 + t] = branchy(t);
    out[64 + t] = fact(t % 7);
}
int main() {
    int* d; float* f;
    cudaMalloc(&d, 128 * sizeof(int));
    cudaMalloc(&f, 32 * sizeof(float));
    k<<<1, 32>>>(d, f);
    cudaDeviceSynchronize();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; }
    int h[128]; float hf[32];
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    cudaMemcpy(hf, f, sizeof hf, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int t = 0; t < 32; ++t) {
        if (h[t] != t + 7) { std::printf("FAIL add[%d]=%d want %d\n", t, h[t], t+7); ++bad; }
        if (hf[t] != t * 2.0f + 1.0f) { std::printf("FAIL scale[%d]=%g\n", t, hf[t]); ++bad; }
        int wb = (t % 2 == 0) ? t * 10 : (t % 3 == 0 ? t * 100 : t);
        if (h[32+t] != wb) { std::printf("FAIL branchy[%d]=%d want %d\n", t, h[32+t], wb); ++bad; }
        int n = t % 7, wf = 1; for (int i = 2; i <= n; ++i) wf *= i;
        if (h[64+t] != wf) { std::printf("FAIL fact[%d]=%d want %d\n", t, h[64+t], wf); ++bad; }
    }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
