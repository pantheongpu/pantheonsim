// __constant__ and __device__ variables reached from the host.
//
// Two halves that only work together. The kernel side needed ld.const, which
// was refused as an unrecognized modifier -- a constant-bank read is a global
// read of a range nothing writes, and the read-only-ness is a promise the
// program makes rather than one this engine enforces.
//
// The host side needed the symbol API, which did not exist: __cudaRegisterVar
// was a deliberate no-op, so cudaMemcpyToSymbol was not even an exported
// symbol and a program calling it failed to load. The handle nvcc passes is
// the address of a host *shadow* object, never a device pointer, so the
// registration is the only thing tying it to a name in the module.
#include <cstdio>
__constant__ float g_coeff[16];
__constant__ int g_scale;
__device__ int g_counter;
__global__ void k(float* out, int* iout) {
    float s = 0;
    for (int i = 0; i < 16; ++i) s += g_coeff[i] * (float)(threadIdx.x + 1);
    out[threadIdx.x] = s * (float)g_scale;
    atomicAdd(&g_counter, 1);
    iout[threadIdx.x] = g_counter;
}
int main() {
    float coeff[16]; for (int i = 0; i < 16; ++i) coeff[i] = (float)(i + 1);
    int scale = 3, zero = 0;
    cudaMemcpyToSymbol(g_coeff, coeff, sizeof coeff);
    cudaMemcpyToSymbol(g_scale, &scale, sizeof scale);
    cudaMemcpyToSymbol(g_counter, &zero, sizeof zero);
    float* f; int* i2;
    cudaMalloc(&f, 32 * sizeof(float)); cudaMalloc(&i2, 32 * sizeof(int));
    k<<<1, 32>>>(f, i2);
    cudaDeviceSynchronize();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(e)); return 1; }
    float hf[32]; cudaMemcpy(hf, f, sizeof hf, cudaMemcpyDeviceToHost);
    int counter = 0; cudaMemcpyFromSymbol(&counter, g_counter, sizeof counter);
    int bad = 0;
    const float coeff_sum = 16.0f * 17.0f / 2.0f;   // 1..16
    for (int t = 0; t < 32; ++t) {
        float want = coeff_sum * (float)(t + 1) * 3.0f;
        if (hf[t] != want) { std::printf("FAIL const[%d]=%g want %g\n", t, hf[t], want); ++bad; }
    }
    if (counter != 32) { std::printf("FAIL device global counter=%d want 32\n", counter); ++bad; }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
