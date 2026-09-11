// Thrust and CUB device-level algorithms, unmodified.
//
// These are the two libraries most CUDA code reaches for, and neither is a
// thin wrapper: Thrust's sort and CUB's radix sort launch tuned multi-kernel
// pipelines with their own temporary storage, decoupled look-back scan, and
// warp-level primitives throughout. Running them end to end exercises far more
// of the engine than a hand-written kernel does, and checks the answer rather
// than that the launch returned.
#include <cstdio>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); ++bad; } } while (0)

int main() {
    int bad = 0;
    const int N = 2048;

    // ---- Thrust ----
    thrust::host_vector<int> h(N);
    long ref = 0;
    for (int i = 0; i < N; ++i) { h[i] = (i * 7919) % 1000; ref += h[i]; }
    thrust::device_vector<int> d = h;
    thrust::sort(d.begin(), d.end());
    const int sum = thrust::reduce(d.begin(), d.end(), 0, thrust::plus<int>());
    thrust::device_vector<int> scan(N);
    thrust::inclusive_scan(d.begin(), d.end(), scan.begin());
    thrust::host_vector<int> sorted = d, hscan = scan;
    if (sum != (int)ref) { std::printf("FAIL thrust::reduce %d want %ld\n", sum, ref); ++bad; }
    for (int i = 1; i < N; ++i)
        if (sorted[i] < sorted[i - 1]) { std::printf("FAIL thrust::sort at %d\n", i); ++bad; break; }
    if (hscan[N - 1] != (int)ref) {
        std::printf("FAIL thrust::inclusive_scan tail %d want %ld\n", (int)hscan[N - 1], ref); ++bad;
    }

    // ---- CUB, device level ----
    int *in = nullptr, *out = nullptr, *dsum = nullptr;
    CK(cudaMalloc(&in, N * sizeof(int)));
    CK(cudaMalloc(&out, N * sizeof(int)));
    CK(cudaMalloc(&dsum, sizeof(int)));
    int raw[N];
    for (int i = 0; i < N; ++i) raw[i] = (int)h[i];
    CK(cudaMemcpy(in, raw, sizeof raw, cudaMemcpyHostToDevice));

    void* tmp = nullptr; size_t tmp_bytes = 0;
    CK(cub::DeviceReduce::Sum(tmp, tmp_bytes, in, dsum, N));
    CK(cudaMalloc(&tmp, tmp_bytes));
    CK(cub::DeviceReduce::Sum(tmp, tmp_bytes, in, dsum, N));
    CK(cudaDeviceSynchronize());
    int hsum = 0; CK(cudaMemcpy(&hsum, dsum, sizeof hsum, cudaMemcpyDeviceToHost));
    if (hsum != (int)ref) { std::printf("FAIL cub DeviceReduce %d want %ld\n", hsum, ref); ++bad; }
    CK(cudaFree(tmp)); tmp = nullptr; tmp_bytes = 0;

    // Decoupled look-back: the scan's later blocks read partial sums the
    // earlier ones publish, so it depends on ordering across blocks and not
    // only on arithmetic.
    CK(cub::DeviceScan::InclusiveSum(tmp, tmp_bytes, in, out, N));
    CK(cudaMalloc(&tmp, tmp_bytes));
    CK(cub::DeviceScan::InclusiveSum(tmp, tmp_bytes, in, out, N));
    CK(cudaDeviceSynchronize());
    int ho[N]; CK(cudaMemcpy(ho, out, sizeof ho, cudaMemcpyDeviceToHost));
    if (ho[N - 1] != (int)ref) {
        std::printf("FAIL cub DeviceScan tail %d want %ld\n", ho[N - 1], ref); ++bad;
    }
    CK(cudaFree(tmp)); tmp = nullptr; tmp_bytes = 0;

    unsigned *kin = nullptr, *kout = nullptr;
    CK(cudaMalloc(&kin, N * sizeof(unsigned)));
    CK(cudaMalloc(&kout, N * sizeof(unsigned)));
    unsigned kh[N];
    for (int i = 0; i < N; ++i) kh[i] = (unsigned)((i * 7919) % 100000);
    CK(cudaMemcpy(kin, kh, sizeof kh, cudaMemcpyHostToDevice));
    CK(cub::DeviceRadixSort::SortKeys(tmp, tmp_bytes, kin, kout, N));
    CK(cudaMalloc(&tmp, tmp_bytes));
    CK(cub::DeviceRadixSort::SortKeys(tmp, tmp_bytes, kin, kout, N));
    CK(cudaDeviceSynchronize());
    unsigned hk[N]; CK(cudaMemcpy(hk, kout, sizeof hk, cudaMemcpyDeviceToHost));
    for (int i = 1; i < N; ++i)
        if (hk[i] < hk[i - 1]) { std::printf("FAIL cub RadixSort at %d\n", i); ++bad; break; }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("cuda error: %s\n", cudaGetErrorString(e)); ++bad; }
    std::printf(bad ? "FAILED\n" : "PASS\n");
    return bad ? 1 : 0;
}
