#include <cstdio>
#include "lib.cuh"
__global__ void k(int* out, float* fout) {
    const int t = threadIdx.x;
    out[t] = biased(t);
    fout[t] = fscale((float)t, 3.0f);
}
int main() {
    int bias[4] = {10, 20, 30, 40};
    cudaMemcpyToSymbol(lib_bias, bias, sizeof bias);
    int* d; float* f;
    cudaMalloc(&d, 32 * sizeof(int)); cudaMalloc(&f, 32 * sizeof(float));
    k<<<1, 32>>>(d, f);
    cudaDeviceSynchronize();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; }
    int h[32]; float hf[32];
    cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    cudaMemcpy(hf, f, sizeof hf, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int t = 0; t < 32; ++t) {
        int want = t * 2 + 1 + bias[t % 4];
        if (h[t] != want) { std::printf("FAIL biased[%d]=%d want %d\n", t, h[t], want); ++bad; }
        float wf = (float)t * 3.0f + 0.5f;
        if (hf[t] != wf) { std::printf("FAIL fscale[%d]=%g want %g\n", t, hf[t], wf); ++bad; }
    }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
