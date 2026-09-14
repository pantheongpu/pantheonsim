// Control-flow differential test: nested divergence, loops with early exit,
// divergent returns, and barriers. VirtualGPU reconverges with a per-warp
// (pc, mask) stack, which is the most approximation-prone part of the engine,
// so these must match a real GPU exactly.
#include <cstdio>
#include <cuda_runtime.h>
#define N 256

__global__ void control(unsigned long long* out) {
    int t = threadIdx.x;
    unsigned long long r = 0;

    // 1. Nested divergence, four distinct paths.
    int a = 0;
    if (t & 1) { if (t & 2) a = 11; else a = 12; }
    else       { if (t & 2) a = 21; else a = 22; }
    r = a;

    // 2. A loop whose trip count varies per lane, with an early break.
    int sum = 0;
    for (int i = 0; i < (t % 7) + 1; ++i) { if (i == 4) break; sum += i * (t + 1); }
    r = r * 1000 + sum;

    // 3. Divergent early return from a helper, expressed inline.
    int z = 0;
    do { if ((t % 5) == 0) { z = 7; break; } if ((t % 5) == 1) { z = 8; break; } z = 9; } while (0);
    r = r * 10 + z;

    // 4. A shared-memory reduction across a barrier, with divergent writers.
    __shared__ int buf[N];
    buf[t] = (t % 3 == 0) ? t : -t;
    __syncthreads();
    int acc = 0;
    if (t == 0) for (int i = 0; i < N; ++i) acc += buf[i];
    __syncthreads();
    // Broadcast through shared memory so every lane observes the same value.
    if (t == 0) buf[0] = acc;
    __syncthreads();
    r = r * 100000 + (unsigned)(buf[0] & 0xFFFF);

    // 5. A deliberately unbalanced loop: lane 0 iterates far more than others.
    int spin = 0;
    for (int i = 0; i < (t == 0 ? 50 : 3); ++i) spin += i;
    r = r * 1000 + spin;

    out[t] = r;
}

int main() {
    unsigned long long *d, h[N];
    if (cudaMalloc(&d, sizeof h) != cudaSuccess) { printf("ALLOCFAIL\n"); return 1; }
    cudaMemset(d, 0, sizeof h);
    control<<<1, N>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("LAUNCHFAIL %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("%d %016llx\n", i, h[i]);
    cudaFree(d);
    return 0;
}
