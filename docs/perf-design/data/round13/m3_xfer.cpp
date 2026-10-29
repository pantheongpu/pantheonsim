// M3 (host-gpu spike): what does moving simulator memory to and from the card cost? The simulator's device memory is 64 KiB chunks,
// each a separate host allocation. Measures, for N MiB: one pageable copy of a contiguous range, one pinned copy, N/64 KiB separate
// pageable copies (one per chunk), gather-to-pinned-staging then one copy, and cuMemHostRegister of the chunks (pinned in place);
// plus the fixed costs of an empty kernel launch + sync, a 4-byte copy and cuMemHostRegister per call.
// Build: g++ -O2 -std=c++17 m3_xfer.cpp -o m3_xfer -I/usr/local/cuda-13.0/include -lcuda
#include <cuda.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <time.h>
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define CK(x) do { CUresult r_ = (x); if (r_) { fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); exit(1); } } while (0)
template <class F> static double best(int reps, F f) { std::vector<double> v; for (int i = 0; i < reps; ++i) { double t = now(); f(); v.push_back(now() - t); } std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
int main() {
  CK(cuInit(0)); CUdevice d; CK(cuDeviceGet(&d, 0)); CUcontext ctx; CK(cuDevicePrimaryCtxRetain(&ctx, d)); CK(cuCtxSetCurrent(ctx));
  const size_t K = 64 << 10; const size_t maxmb = 1024;
  CUdeviceptr dev; CK(cuMemAlloc(&dev, maxmb << 20));
  // chunks as the simulator has them: separate 64 KiB heap allocations (zero-filled then touched)
  std::vector<uint8_t*> chunks(maxmb * 16); for (auto& c : chunks) { c = new uint8_t[K]; memset(c, 0x5a, K); }
  std::vector<uint8_t> contig(maxmb << 20, 0x3c);
  void* pin; CK(cuMemAllocHost(&pin, maxmb << 20)); memset(pin, 1, maxmb << 20);
  void* stage; CK(cuMemAllocHost(&stage, maxmb << 20));
  // fixed costs
  { const char* ptx = ".version 7.0\n.target sm_75\n.address_size 64\n.visible .entry empty(){ ret; }\n"; CUmodule m; CK(cuModuleLoadData(&m, ptx)); CUfunction f; CK(cuModuleGetFunction(&f, m, "empty"));
    for (int i = 0; i < 50; ++i) { cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, 0, 0, 0); cuCtxSynchronize(); }
    double t = best(200, [&] { cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, 0, 0, 0); cuCtxSynchronize(); }); printf("empty kernel launch + sync: %.1f us\n", t * 1e6);
    t = best(200, [&] { cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, 0, 0, 0); }); cuCtxSynchronize(); printf("empty kernel launch (async submit only): %.1f us\n", t * 1e6);
    int x = 5; t = best(200, [&] { cuMemcpyHtoD(dev, &x, 4); }); printf("4-byte HtoD: %.1f us\n", t * 1e6); t = best(200, [&] { cuMemcpyDtoH(&x, dev, 4); }); printf("4-byte DtoH: %.1f us\n", t * 1e6);
    t = best(50, [&] { cuMemcpyHtoD(dev, contig.data(), K); }); printf("64 KiB pageable HtoD (one chunk): %.1f us\n", t * 1e6);
    t = best(50, [&] { cuMemcpyDtoH(contig.data(), dev, K); }); printf("64 KiB pageable DtoH (one chunk): %.1f us\n", t * 1e6);
  }
  printf("%6s | %8s %8s | %8s %8s | %10s %10s | %10s %10s | %10s\n", "MiB", "pgbl H2D", "pgbl D2H", "pin H2D", "pin D2H", "chunk H2D", "chunk D2H", "gather+pin", "pin+scatter", "register+H2D");
  for (size_t mb : {1, 4, 16, 64, 256, 1024}) {
    size_t n = mb << 20, nc = n / K; int reps = mb >= 256 ? 3 : 7;
    double a = best(reps, [&] { CK(cuMemcpyHtoD(dev, contig.data(), n)); });
    double b = best(reps, [&] { CK(cuMemcpyDtoH(contig.data(), dev, n)); });
    double c = best(reps, [&] { CK(cuMemcpyHtoD(dev, pin, n)); });
    double e = best(reps, [&] { CK(cuMemcpyDtoH(pin, dev, n)); });
    double f = best(reps, [&] { for (size_t i = 0; i < nc; ++i) CK(cuMemcpyHtoD(dev + i * K, chunks[i], K)); });
    double g = best(reps, [&] { for (size_t i = 0; i < nc; ++i) CK(cuMemcpyDtoH(chunks[i], dev + i * K, K)); });
    double h = best(reps, [&] { for (size_t i = 0; i < nc; ++i) memcpy((uint8_t*)stage + i * K, chunks[i], K); CK(cuMemcpyHtoD(dev, stage, n)); });
    double i2 = best(reps, [&] { CK(cuMemcpyDtoH(stage, dev, n)); for (size_t i = 0; i < nc; ++i) memcpy(chunks[i], (uint8_t*)stage + i * K, K); });
    double r = best(reps, [&] { for (size_t i = 0; i < nc; ++i) CK(cuMemHostRegister(chunks[i], K, 0)); for (size_t i = 0; i < nc; ++i) CK(cuMemcpyHtoD(dev + i * K, chunks[i], K)); for (size_t i = 0; i < nc; ++i) CK(cuMemHostUnregister(chunks[i])); });
    auto gbs = [&](double t) { return n / t / 1e9; };
    printf("%6zu | %5.1f GB/s %5.1f | %5.1f %5.1f | %6.2f %6.2f GB/s | %6.2f %6.2f | %6.2f GB/s (%.1f ms)\n", mb, gbs(a), gbs(b), gbs(c), gbs(e), gbs(f), gbs(g), gbs(h), gbs(i2), gbs(r), r * 1e3);
  }
  // fixed per-chunk costs, to separate latency from bandwidth
  double t = best(7, [&] { for (int i = 0; i < 1024; ++i) CK(cuMemcpyHtoD(dev + i * K, chunks[i], K)); }); printf("per 64 KiB pageable chunk HtoD: %.1f us (%.2f GB/s)\n", t / 1024 * 1e6, 1024 * K / t / 1e9);
  t = best(7, [&] { for (int i = 0; i < 1024; ++i) CK(cuMemHostRegister(chunks[i], K, 0)); for (int i = 0; i < 1024; ++i) CK(cuMemHostUnregister(chunks[i])); }); printf("per chunk cuMemHostRegister+Unregister: %.1f us\n", t / 1024 * 1e6);
  return 0;
}
