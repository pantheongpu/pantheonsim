// The device runtime's cudaGetDevice and cudaGetDeviceCount, called from a
// kernel. They need separate compilation (run_rdc_link.sh builds with
// -rdc=true), and reach the driver through __cuda_syscall_cnpv2GetDevice and
// ...GetDeviceCount, which the simulator used to refuse as unknown calls.
// NanoVDB's tests call cudaGetDevice from a kernel. Each answer is checked
// against the host's own.
#include <cstdio>

__global__ void ask(int* out) {
    int dev = -1, count = -1;
    out[0] = (int)cudaGetDevice(&dev);
    out[1] = dev;
    out[2] = (int)cudaGetDeviceCount(&count);
    out[3] = count;
}

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    int bad = 0;
    for (int d = 0; d < count; ++d) {
        cudaSetDevice(d);
        int* out = nullptr;
        int h[4] = {-9, -9, -9, -9};
        cudaMalloc(&out, sizeof h);
        ask<<<1, 1>>>(out);
        if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL launch on device %d\n", d); ++bad; continue; }
        cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost);
        if (h[0] != 0 || h[1] != d) { std::printf("FAIL device %d: cudaGetDevice -> err %d, %d\n", d, h[0], h[1]); ++bad; }
        if (h[2] != 0 || h[3] != count) { std::printf("FAIL device %d: cudaGetDeviceCount -> err %d, %d (host says %d)\n", d, h[2], h[3], count); ++bad; }
        cudaFree(out);
    }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
