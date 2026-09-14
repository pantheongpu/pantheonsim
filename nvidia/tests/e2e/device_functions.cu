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

// Structs by value, in both directions. A call slot is a byte buffer per lane,
// so a 16-byte struct is as much a slot as an int is -- but it only became one
// when the slot stopped being a single value per lane.
// Indirect calls. The table is an array global initialised with a list of
// symbols, so it exercises three things at once: functions having addresses at
// all, an array initialiser that is a list of names rather than numbers, and
// a call whose target is only known at execution.
typedef int (*binop)(int, int);
__device__ int d_add(int a, int b) { return a + b; }
__device__ int d_mul(int a, int b) { return a * b; }
__device__ int d_sub(int a, int b) { return a - b; }
__device__ binop d_table[3] = {d_add, d_mul, d_sub};

struct Big { float a, b, c, d; };
__device__ __noinline__ Big make_big(float x) { return Big{x, x + 1, x + 2, x + 3}; }
__device__ __noinline__ float consume(Big b) { return b.a + b.b * 2 + b.c * 3 + b.d * 4; }

__global__ void k(int* out, float* fout, int which) {
    const int t = threadIdx.x;
    { binop f = d_table[which]; out[96 + t] = f(t + 2, 3); }
    out[t] = slow_add(t, 7);
    fout[t] = slow_scale((float)t, 2.0f);
    out[32 + t] = branchy(t);
    out[64 + t] = fact(t % 7);
    Big b = make_big((float)t);
    fout[32 + t] = b.a + b.b + b.c + b.d;
    fout[64 + t] = consume(b);
}
int main() {
    int* d; float* f;
    cudaMalloc(&d, 128 * sizeof(int));
    cudaMalloc(&f, 128 * sizeof(float));
    k<<<1, 32>>>(d, f, 1);   // d_mul
    cudaDeviceSynchronize();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; }
    int h[128]; float hf[128];
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
        float x = (float)t;
        float sum = x + (x+1) + (x+2) + (x+3);
        float con = x + (x+1)*2 + (x+2)*3 + (x+3)*4;
        if (hf[32+t] != sum) { std::printf("FAIL struct ret[%d]=%g want %g\n", t, hf[32+t], sum); ++bad; }
        if (hf[64+t] != con) { std::printf("FAIL struct arg[%d]=%g want %g\n", t, hf[64+t], con); ++bad; }
        if (h[96+t] != (t + 2) * 3) {
            std::printf("FAIL funcptr[%d]=%d want %d\n", t, h[96+t], (t+2)*3); ++bad;
        }
    }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
