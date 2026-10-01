// Static __shared__ arrays inside device functions, built with -rdc=true
// (run_rdc_link.sh). Separate compilation hoists them to module scope, and a
// kernel must be charged only for the ones it can reach: each kernel here
// needs 40 KB, the two together 80 KB, more than a multiprocessor holds.
// Charging every kernel for every module-scope array refused both launches
// (NanoVDB's tests, with the device-runtime library's arrays on top, could
// launch nothing). Built normally, this is just a correct program.
#include <cstdio>

__device__ __noinline__ int stage_a(int i) {
    __shared__ int buf[10000];
    buf[i] = i + 1;
    __syncthreads();
    return buf[(i + 1) % blockDim.x];
}
__device__ __noinline__ int stage_b(int i) {
    __shared__ int buf[10000];
    buf[i] = 2 * i;
    __syncthreads();
    return buf[(i + 1) % blockDim.x];
}
__global__ void kernel_a(int* out) { out[threadIdx.x] = stage_a(threadIdx.x); }
__global__ void kernel_b(int* out) { out[32 + threadIdx.x] = stage_b(threadIdx.x); }

int main() {
    int* d = nullptr;
    int h[64] = {0};
    cudaMalloc(&d, sizeof h);
    kernel_a<<<1, 32>>>(d);
    const cudaError_t ea = cudaDeviceSynchronize();
    kernel_b<<<1, 32>>>(d);
    const cudaError_t eb = cudaDeviceSynchronize();
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int bad = 0;
    if (ea != cudaSuccess) { std::printf("FAIL kernel_a: %s\n", cudaGetErrorString(ea)); ++bad; }
    if (eb != cudaSuccess) { std::printf("FAIL kernel_b: %s\n", cudaGetErrorString(eb)); ++bad; }
    for (int t = 0; t < 32 && !bad; ++t) {
        const int n = (t + 1) % 32;
        if (h[t] != n + 1) { std::printf("FAIL kernel_a[%d]=%d want %d\n", t, h[t], n + 1); ++bad; }
        if (h[32 + t] != 2 * n) { std::printf("FAIL kernel_b[%d]=%d want %d\n", t, h[32 + t], 2 * n); ++bad; }
    }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
