// String literals read inside non-inlined device functions.
//
// Built with -rdc=true (run_rdc_link.sh), a program is linked together with
// CUDA's device-runtime library, and both name their string literals $str,
// $str$1, ... Those names are file-local, but when the pieces were pasted into
// one module the library's definitions won, so a literal read "cudaSuccess"
// or "no error". NanoVDB's device strcmp returned 17 for two equal strings
// ('t' - 'c'). Built normally, this is just a correct program.
#include <cstdio>

__device__ __noinline__ int dev_strcmp(const char* lhs, const char* rhs) {
    while (*lhs != '\0' && *lhs == *rhs) { ++lhs; ++rhs; }
    return *(const unsigned char*)lhs - *(const unsigned char*)rhs;
}
__device__ __noinline__ int first_byte(const char* p) { return *(const unsigned char*)p; }

__global__ void check(const char* buf, int* out) {
    out[0] = first_byte("this is a test #2");             // 't' = 116
    out[1] = dev_strcmp(buf, "this is a test #2");         // equal: 0
    out[2] = dev_strcmp("this is a test #2", buf);         // equal: 0
    out[3] = dev_strcmp(buf, "this is a test");            // ' ' - '\0' = 32
    out[4] = dev_strcmp("abc", "abd");                     // 'c' - 'd' = -1
    out[5] = first_byte("no error");                       // 'n' = 110, the library's $str$1
}

int main() {
    const char text[] = "this is a test #2";
    char* buf = nullptr;
    int* out = nullptr;
    cudaMalloc(&buf, sizeof text);
    cudaMalloc(&out, 6 * sizeof(int));
    cudaMemcpy(buf, text, sizeof text, cudaMemcpyHostToDevice);
    check<<<1, 1>>>(buf, out);
    int h[6] = {0};
    if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAILED: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    const int want[6] = {116, 0, 0, 32, -1, 110};
    const char* what[6] = {"first byte of a literal", "buffer vs equal literal", "literal vs equal buffer",
                           "buffer vs its prefix", "two literals", "a second literal"};
    int bad = 0;
    for (int i = 0; i < 6; ++i)
        if (h[i] != want[i]) { std::printf("FAIL %s: %d, want %d\n", what[i], h[i], want[i]); ++bad; }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
